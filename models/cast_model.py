import itertools
import torch
from .base_model import BaseModel
from . import net
import util.util as util
from util.image_pool import ImagePool
import torch.nn as nn
from torch.nn import init

class CASTModel(BaseModel):
    """ This class implements CAST model.
    This code is inspired by DCLGAN and CycleGAN.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CAST """
        parser.add_argument('--CAST_mode', type=str, default="CAST", choices='CAST')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CAST.
        if opt.CAST_mode.lower() == "cast":
            pass
            # parser.set_defaults(nce_idt=False, lambda_NCE=2.0)
        else:
            raise ValueError(opt.CAST_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.model_names = ['AE','Dec_A', 'Dec_B']

        # define networks (both generator and discriminator)
        
        vgg = net.vgg
        vgg = nn.Sequential(*list(vgg.children())[:31]) 
        self.netAE = net.ADAIN_Encoder(vgg, self.gpu_ids)
        self.netDec_A = net.Decoder(self.gpu_ids)
        self.netDec_B = net.Decoder(self.gpu_ids)
        self.visual_names = ['fake_B']
        
    def training(self):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass
        
    def train(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass
    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
            

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
        self.real_B_feat = self.netAE(self.real_B, self.real_A)  # G_A(A)
        self.fake_A = self.netDec_A(self.real_B_feat)
        self.fake_B = self.netDec_B(self.real_A_feat)
        
    def generate_visuals_for_evaluation(self, data, mode):
        with torch.no_grad():
            visuals = {}
            AtoB = self.opt.direction == "AtoB"
            G = self.netG
            source = data["A" if AtoB else "B"].to(self.device)
            if mode == "forward":
                visuals["fake_B"] = G(source)
            else:
                raise ValueError("mode %s is not recognized" % mode)
            return visuals
    
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
