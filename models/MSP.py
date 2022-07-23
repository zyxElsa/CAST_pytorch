import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from .torch_utils import concat_all_gather, get_world_size


class StyleExtractor(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, encoder, gpu_ids = []):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(StyleExtractor, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:6])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[6:13])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[13:20])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[20:33])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[33:46])  # relu4_1 -> relu5_1
        self.enc_6 = nn.Sequential(*enc_layers[46:70])  # relu5_1 -> maxpool

        # fix the encoder
        for name in ['enc_1', 'enc_2','enc_3', 'enc_4', 'enc_5', 'enc_6']:
            for param in getattr(self, name).parameters():
                param.requires_grad = True

        # Class Activation Map
        # self.gap_fc0 = nn.Linear(64, 1, bias=False)
        # self.gmp_fc0 = nn.Linear(64, 1, bias=False)
        # self.gap_fc1 = nn.Linear(128, 1, bias=False)
        # self.gmp_fc1 = nn.Linear(128, 1, bias=False)
        # self.gap_fc2 = nn.Linear(256, 1, bias=False)
        # self.gmp_fc2 = nn.Linear(256, 1, bias=False)
        # self.gap_fc3 = nn.Linear(512, 1, bias=False)
        # self.gmp_fc3 = nn.Linear(512, 1, bias=False)
        # self.gap_fc4 = nn.Linear(512, 1, bias=False)
        # self.gmp_fc4 = nn.Linear(512, 1, bias=False)
        # self.gap_fc5 = nn.Linear(512, 1, bias=False)
        # self.gmp_fc5 = nn.Linear(512, 1, bias=False)
        self.conv1x1_0 = nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=True)
        self.conv1x1_1 = nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=True)
        self.conv1x1_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=True)
        self.conv1x1_3 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.conv1x1_4 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.conv1x1_5 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(6):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def forward(self, input, index):
        """Standard forward."""
        feats = self.encode_with_intermediate(input)
        codes = []
        for x in index:
            code = feats[x].clone()
            gap = torch.nn.functional.adaptive_avg_pool2d(code, (1,1))
            gmp = torch.nn.functional.adaptive_max_pool2d(code, (1,1))            
            conv1x1 = getattr(self, 'conv1x1_{:d}'.format(x))
            code = torch.cat([gap, gmp], 1)
            code = self.relu(conv1x1(code))
            codes.append(code)
        return codes 



class Projector(nn.Module):
    def __init__(self, projector, gpu_ids = []):
        super(Projector, self).__init__()
        self.projector0 = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector1 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(128, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector2 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(256,1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector3 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector4 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )
        self.projector5 = nn.Sequential(
            #nn.Dropout(),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
        )

    def forward(self, input, index):
        """Standard forward."""
        num = 0
        projections = []
        for x in index:
            projector = getattr(self, 'projector{:d}'.format(x))        
            code = input[num].view(input[num].size(0), -1)
            projection = projector(code).view(code.size(0), -1)
            projection = nn.functional.normalize(projection)
            projections.append(projection)
            num += 1
        return projections


def make_layers(cfg, batch_norm=True):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

vgg = make_layers([3, 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M', 512, 512, 'M', 512, 512, 'M'])

class InfoNCELoss(nn.Module):

    def __init__(self, temperature, feature_dim, queue_size):
        super().__init__()
        self.tau = temperature
        self.queue_size = queue_size
        self.world_size = get_world_size()
        data0 = torch.randn(2048, queue_size)
        data0 = F.normalize(data0, dim=0)
        data1 = torch.randn(2048, queue_size)
        data1 = F.normalize(data1, dim=0)
        data2 = torch.randn(2048, queue_size)
        data2 = F.normalize(data2, dim=0)
        data3 = torch.randn(2048, queue_size)
        data3 = F.normalize(data3, dim=0)
        data4 = torch.randn(2048, queue_size)
        data4 = F.normalize(data4, dim=0)
        data5 = torch.randn(2048, queue_size)
        data5 = F.normalize(data5, dim=0)
        
        self.register_buffer("queue_data_A0", data0)
        self.register_buffer("queue_ptr_A0", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_B0", data0)
        self.register_buffer("queue_ptr_B0", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_data_A2", data2)
        self.register_buffer("queue_ptr_A2", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_B2", data2)
        self.register_buffer("queue_ptr_B2", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("queue_data_A4", data4)
        self.register_buffer("queue_ptr_A4", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_B4", data4)
        self.register_buffer("queue_ptr_B4", torch.zeros(1, dtype=torch.long))
        
        self.register_buffer("queue_data_A1", data1)
        self.register_buffer("queue_ptr_A1", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_B1", data1)
        self.register_buffer("queue_ptr_B1", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_data_A3", data3)
        self.register_buffer("queue_ptr_A3", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_B3", data3)
        self.register_buffer("queue_ptr_B3", torch.zeros(1, dtype=torch.long))        

        self.register_buffer("queue_data_A5", data5)
        self.register_buffer("queue_ptr_A5", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_data_B5", data5)
        self.register_buffer("queue_ptr_B5", torch.zeros(1, dtype=torch.long))

    def forward(self, query, key, style = 'real'):
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", (query, key)).unsqueeze(-1)

        # negative logits: NxK
        if style == 'real_A0':
            queue = self.queue_data_A0.clone().detach()        
        elif style == 'real_A1':
            queue = self.queue_data_A1.clone().detach()
        elif style == 'real_A2':
            queue = self.queue_data_A2.clone().detach()
        elif style == 'real_A3':
            queue = self.queue_data_A3.clone().detach()
        elif style == 'real_A4':
            queue = self.queue_data_A4.clone().detach()
        elif style == 'real_A5':
            queue = self.queue_data_A5.clone().detach()
        elif style == 'fake_A':
            queue = self.queue_data_fake_A.clone().detach()  
        elif style == 'real_B0':
            queue = self.queue_data_B0.clone().detach()
        elif style == 'real_B1':
            queue = self.queue_data_B1.clone().detach()
        elif style == 'real_B2':
            queue = self.queue_data_B2.clone().detach()
        elif style == 'real_B3':
            queue = self.queue_data_B3.clone().detach()
        elif style == 'real_B4':
            queue = self.queue_data_B4.clone().detach()
        elif style == 'real_B5':
            queue = self.queue_data_B5.clone().detach()
        elif style == 'fake_B':
            queue = self.queue_data_fake_B.clone().detach()           
        else:
            raise NotImplementedError('QUEUE: style is not recognized')
        l_neg = torch.einsum("nc,ck->nk", (query, queue))

        # logits: Nx(1+K)
        logits = torch.cat((l_pos, l_neg), dim=1)

        # labels: positive key indicators
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=query.device)

        return F.cross_entropy(logits / self.tau, labels)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, style = 'real'):
        # gather from all gpus
        if self.world_size > 1:
            keys = concat_all_gather(keys, self.world_size)
        batch_size = keys.size(0)
        # replace the keys at ptr (dequeue and enqueue)
        if style == 'real_A0':
            ptr = int(self.queue_ptr_A0)
            assert self.queue_size % batch_size == 0
            self.queue_data_A0[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A0[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_A1':
            ptr = int(self.queue_ptr_A1)
            assert self.queue_size % batch_size == 0
            self.queue_data_A1[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A1[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_A2':
            ptr = int(self.queue_ptr_A2)
            assert self.queue_size % batch_size == 0
            self.queue_data_A2[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A2[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_A3':
            ptr = int(self.queue_ptr_A3)
            assert self.queue_size % batch_size == 0
            self.queue_data_A3[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A3[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_A4':
            ptr = int(self.queue_ptr_A4)
            assert self.queue_size % batch_size == 0
            self.queue_data_A4[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A4[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_A5':
            ptr = int(self.queue_ptr_A5)
            assert self.queue_size % batch_size == 0
            self.queue_data_A5[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_A5[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_B0':
            ptr = int(self.queue_ptr_B0)
            assert self.queue_size % batch_size == 0
            self.queue_data_B0[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_B0[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_B1':
            ptr = int(self.queue_ptr_B1)
            assert self.queue_size % batch_size == 0
            self.queue_data_B1[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_B1[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_B2':
            ptr = int(self.queue_ptr_B2)
            assert self.queue_size % batch_size == 0
            self.queue_data_B2[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_B2[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_B3':
            ptr = int(self.queue_ptr_B3)
            assert self.queue_size % batch_size == 0
            self.queue_data_B3[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_B3[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_B4':
            ptr = int(self.queue_ptr_B4)
            assert self.queue_size % batch_size == 0
            self.queue_data_B4[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_B4[0] = (ptr + batch_size) % self.queue_size
        elif style == 'real_B5':
            ptr = int(self.queue_ptr_B5)
            assert self.queue_size % batch_size == 0
            self.queue_data_B5[:, ptr:ptr + batch_size] = keys.T
            self.queue_ptr_B5[0] = (ptr + batch_size) % self.queue_size
        else:
            raise NotImplementedError('QUEUE: style is not recognized')

