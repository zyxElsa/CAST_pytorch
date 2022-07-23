import itertools
import torch
from .base_model import BaseModel
from . import networks
from . import net
from . import MSP
import util.util as util
from util.image_pool import ImagePool
import torch.nn as nn
from torch.nn import init
import kornia.augmentation as K

class CASTModel(BaseModel):
    """ This class implements CAST model.
    This code is inspired by DCLGAN
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CAST """
        parser.add_argument('--CAST_mode', type=str, default="CAST", choices='CAST')
        parser.add_argument('--lambda_GAN_G_A', type=float, default=0.1, help='weight for GAN loss：GAN(G(Ic, Is))')
        parser.add_argument('--lambda_GAN_G_B', type=float, default=0.1, help='weight for GAN loss：GAN(G(Is, Ic))')

        parser.add_argument('--lambda_GAN_D_A', type=float, default=1.0, help='weight for GAN loss：GAN(G(Is, Ic))')
        parser.add_argument('--lambda_GAN_D_B', type=float, default=1.0, help='weight for GAN loss：GAN(G(Ic, Is))')
        
        parser.add_argument('--lambda_NCE_G', type=float, default=0.05, help='weight for NCE loss: NCE(G(Ic, Is), Is)')
        parser.add_argument('--lambda_NCE_D', type=float, default=1.0, help='weight for NCE loss: NCE(I, I+, I-)')
        
        parser.add_argument('--lambda_CYC', type=float, default=4.0, help='weight for l1 reconstructe loss:||Ic - G(G(Ic, Is),Ic)||')
        
        parser.add_argument('--nce_layers', type=str, default='0,1,2,3', help='compute NCE loss on which layers')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CAST.
        if opt.CAST_mode.lower() == "cast":
            pass
        else:
            raise ValueError(opt.CAST_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        

        if self.opt.lambda_GAN_G_A > 0.0 and self.isTrain:
            self.loss_names += [ 'G_A']
        if self.opt.lambda_GAN_G_B > 0.0 and self.isTrain:
            self.loss_names += [ 'G_B']

        if self.opt.lambda_GAN_D_A > 0.0 and self.isTrain:
            self.loss_names += ['D_A']
        if self.opt.lambda_GAN_D_B > 0.0 and self.isTrain:
            self.loss_names += ['D_B']

        if self.opt.lambda_NCE_G > 0.0 and self.isTrain:
            self.loss_names += [ 'G_NCE_style']

        if self.opt.lambda_NCE_D > 0.0 and self.isTrain:
            self.loss_names += [ 'NCE_D']

        if self.opt.lambda_CYC > 0.0 and self.isTrain:
            self.visual_names += ['rec_A', 'rec_B']
            self.loss_names += ['cyc']

        if self.isTrain:
            self.model_names = ['AE','Dec_A', 'Dec_B', 'D', 'P_style', 'D_A', 'D_B']
        else:  # during test time, only load G
            self.model_names = ['AE','Dec_A', 'Dec_B']

        # define networks 
        
        vgg = net.vgg
        vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
        vgg = nn.Sequential(*list(vgg.children())[:31]) 
        self.netAE = net.ADAIN_Encoder(vgg, self.gpu_ids)
        self.netDec_A = net.Decoder(self.gpu_ids)
        self.netDec_B = net.Decoder(self.gpu_ids)  
        init_net(self.netAE, 'normal', 0.02, self.gpu_ids)  
        init_net(self.netDec_A, 'normal', 0.02, self.gpu_ids)  
        init_net(self.netDec_B, 'normal', 0.02, self.gpu_ids)

        if self.isTrain:       
            style_vgg = MSP.vgg
            style_vgg.load_state_dict(torch.load('models/style_vgg.pth'))
            style_vgg = nn.Sequential(*list(style_vgg.children()))
            self.netD = MSP.StyleExtractor(style_vgg, self.gpu_ids)  
            self.netP_style = MSP.Projector(self.gpu_ids)  
            init_net(self.netD, 'normal', 0.02, self.gpu_ids) 
            init_net(self.netP_style, 'normal', 0.02, self.gpu_ids)
            
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                            opt.crop_size, opt.feature_dim, opt.max_conv_dim,
                                            opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                            opt.crop_size, opt.feature_dim, opt.max_conv_dim,
                                            opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                            self.gpu_ids, opt)        

            self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

            self.nce_loss = MSP.InfoNCELoss(opt.temperature, opt.hypersphere_dim, 
                                             opt.queue_size).to(self.device)
            self.mse_loss = nn.MSELoss()
            
            self.patch_sampler = K.RandomResizedCrop((256,256),scale=(0.8,1.0),ratio=(0.75,1.33)).to(self.device)
            
            self.criterionCyc = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netAE.parameters(), self.netDec_A.parameters(), self.netDec_B.parameters()),
                                                lr=opt.lr_G, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.optimizer_D_NCE = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netP_style.parameters()),
                                                lr=opt.lr_D_NCE, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_NCE)


    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        if self.opt.lambda_GAN_D_A > 0.0 or self.opt.lambda_GAN_D_B > 0.0:
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.set_requires_grad([self.netD, self.netP_style, self.netAE, self.netDec_A,self.netDec_B ], False)
            self.optimizer_D.zero_grad()
            self.loss_D = self.backward_D()
            self.loss_D.backward(retain_graph=True)
            self.optimizer_D.step()
        
        # update MSP
        if self.opt.lambda_NCE_D > 0.0:
            self.set_requires_grad([self.netD, self.netP_style], True)
            self.set_requires_grad([self.netAE, self.netDec_A,self.netDec_B, self.netD_A, self.netD_B ], False)
            self.optimizer_D_NCE.zero_grad()
            self.loss_NCE_D = self.backward_D_NCEloss()
            self.loss_NCE_D.backward(retain_graph=True)
            self.optimizer_D_NCE.step()

        # update G
        self.set_requires_grad([self.netD, self.netP_style, self.netD_A, self.netD_B], False)
        self.set_requires_grad([self.netAE, self.netDec_A,self.netDec_B], True)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
            

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.real_A_feat = self.netAE(self.real_A, self.real_B)  # G_A(A)
        self.fake_B = self.netDec_B(self.real_A_feat)
        if self.isTrain: 
            self.real_B_feat = self.netAE(self.real_B, self.real_A)  # G_A(A)
            self.fake_A = self.netDec_A(self.real_B_feat)
            if self.opt.lambda_CYC > 0.0:
                self.rec_A_feat = self.netAE(self.fake_B, self.real_A)
                self.rec_B_feat = self.netAE(self.fake_A, self.real_B)
                self.rec_A = self.netDec_A(self.rec_A_feat)
                self.rec_B = self.netDec_B(self.rec_B_feat)

    def backward_D_basic(self, netD, content,style, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
       
        loss_D_real = loss_D_fake = 0
        # Real
        pred_real = netD(style)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake)*0.5        
        return loss_D
        
    def backward_D_NCEloss(self):
        """
        Calculate NCE loss for the discriminator
        """
        #query_A = query_B =0.0
        real_A = self.netD(self.patch_sampler(self.real_A), self.nce_layers)
        real_B = self.netD(self.patch_sampler(self.real_B), self.nce_layers)
        real_Ax = self.netD(self.patch_sampler(self.real_A), self.nce_layers)
        real_Bx = self.netD(self.patch_sampler(self.real_B), self.nce_layers)

        query_A = self.netP_style(real_A, self.nce_layers)
        query_B = self.netP_style(real_B, self.nce_layers)
        query_Ax = self.netP_style(real_Ax, self.nce_layers)  
        query_Bx = self.netP_style(real_Bx, self.nce_layers) 

        num = 0
        loss_D_cont_A = 0
        loss_D_cont_B = 0
        for x in self.nce_layers:
            #self.nce_loss.dequeue_and_enqueue(query_A[num], 'real_A{:d}'.format(x))
            self.nce_loss.dequeue_and_enqueue(query_B[num], 'real_B{:d}'.format(x))
            #loss_D_cont_A += self.nce_loss(query_A[num], query_Ax[num], 'real_B{:d}'.format(x))
            loss_D_cont_B += self.nce_loss(query_B[num], query_Bx[num], 'real_B{:d}'.format(x))
            num += 1
        
        loss_NCE_D  = (loss_D_cont_A + loss_D_cont_B) * 0.5 * self.opt.lambda_NCE_D
        return loss_NCE_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        if self.opt.lambda_GAN_D_B > 0.0:
            fake_B = self.fake_pool.query(self.fake_B)
            self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.real_B, fake_B) * self.opt.lambda_GAN_D_B
        else:
            self.loss_D_B = 0

        if self.opt.lambda_GAN_D_A > 0.0:
            fake_A = self.fake_pool.query(self.fake_A)
            self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.real_A, fake_A) * self.opt.lambda_GAN_D_A

        else:
            self.loss_D_A = 0

        self.loss_D = (self.loss_D_B + self.loss_D_A) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN_G_A > 0.0:
            pred_fakeB = self.netD_B(self.fake_B)
            self.loss_G_A = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN_G_A
        else:
            self.loss_G_A = 0.0

        if self.opt.lambda_GAN_G_B > 0.0:
            pred_fakeA = self.netD_A(self.fake_A)
            self.loss_G_B = self.criterionGAN(pred_fakeA, True).mean() * self.opt.lambda_GAN_G_B
        else:
            self.loss_G_B = 0.0

        # Calculate the style contrastive loss.
        if self.opt.lambda_NCE_G > 0.0:
            real_A = self.patch_sampler(self.real_A)
            real_B = self.patch_sampler(self.real_B)
            fake_A = self.patch_sampler(self.fake_A)
            fake_B = self.patch_sampler(self.fake_B)

            key_A = self.netP_style(self.netD(real_A, self.nce_layers),self.nce_layers)
            key_B = self.netP_style(self.netD(real_B, self.nce_layers),self.nce_layers)
            query_A = self.netP_style(self.netD(fake_A, self.nce_layers),self.nce_layers)
            query_B = self.netP_style(self.netD(fake_B, self.nce_layers),self.nce_layers)

            num = 0
            self.loss_G_NCE_style_A = 0
            self.loss_G_NCE_style_B = 0
            for x in self.nce_layers:
                #self.loss_G_NCE_style_A += self.nce_loss(query_A[num], key_A[num], 'real_B{:d}'.format(x))
                self.loss_G_NCE_style_B += self.nce_loss(query_B[num], key_B[num], 'real_B{:d}'.format(x))
                num += 1
        else:
            self.loss_G_NCE_style_A = 0
            self.loss_G_NCE_style_B = 0
        self.loss_G_NCE_style = (self.loss_G_NCE_style_A + self.loss_G_NCE_style_B) * 0.5 * self.opt.lambda_NCE_G
        
        #L1 Cycle Loss
        if self.opt.lambda_CYC > 0.0:
            self.loss_cyc_A = self.criterionCyc(self.rec_A, self.real_A) * self.opt.lambda_CYC
            self.loss_cyc_B = self.criterionCyc(self.rec_B, self.real_B) * self.opt.lambda_CYC
        else:
            self.loss_cyc_A = 0
            self.loss_cyc_B = 0
        self.loss_cyc = (self.loss_cyc_A + self.loss_cyc_B) * 0.5

        self.loss_G = self.loss_cyc + self.loss_G_NCE_style + (self.loss_G_A + self.loss_G_B) * 0.5
        
        return self.loss_G


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
