import torch.nn as nn
import torch
import numpy as np
import copy

from .convlstm import *
from .submodule import *
from .vit_pytorch import *
#from module import *

class Transformer_FeatNet(nn.Module):
    def __init__(self, input_size, hidden_dim=768):
        super(Transformer_FeatNet, self).__init__()
        
        self.in_planes = hidden_dim
        self.base = vit_small_patch16_224_TransReID(img_size=input_size, aie_xishu=2.5,local_feature=False, camera=0, view=0, stride_size=(16,16), drop_path_rate=0.1)
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
    
    def forward(self, x):  # label is unused if self.cos_layer == 'no'
        global_feat = self.base(x)
        feat = global_feat[0]
        if not feat.shape[0] == 1:
            feat = self.bottleneck(feat)
        return feat

class Transformer_CostNet(nn.Module):
    def __init__(self, hidden_dim=768):
        super(Transformer_CostNet, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    
    def forward(self, x):  # label is unused if self.cos_layer == 'no'
        feat = self.transformer_encoder(x)
        return feat

class DecoderNet(nn.Module):
    def __init__(self, input_size, hidden_dim=768, bias=False):
        super(DecoderNet, self).__init__()
        self.bias = bias
        self.h, self.w = input_size
        self.linear = nn.Linear(hidden_dim, (self.h * self.w) // 256)
        self.linear2 = nn.Linear(64, 1)

        self.conv0 = convgnrelu(8, 8, kernel_size=3, stride=1, dilation=1)
        self.conv1 = convgnrelu(16, 16, kernel_size=3, stride=1, dilation=1)
        self.conv2 = convgnrelu(32, 32, kernel_size=3, stride=1, dilation=1)
        self.conv3 = convgnrelu(64, 64, kernel_size=3, stride=1, dilation=1)
    
        self.deconv_0 = deConvGnReLU(
            1,
            8, #16
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
        )

        self.deconv_1 = deConvGnReLU(
            8,
            16, #16
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
        )

        self.deconv_2 = deConvGnReLU(
            16,
            32, #16
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
        )

        self.deconv_3 = deConvGnReLU(
            32,
            64, #16
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
        )

    def forward(self, x):  # label is unused if self.cos_layer == 'no'
        x = self.linear(x) # [BD, 768] => [BD, H*W/256]
        x = x.reshape(x.shape[0], 1, self.h // 16, self.w // 16) # [BD, H*W/256] => [BD, 1, H/16, W/16]
        x1 = self.conv0(self.deconv_0(x)) # [BD, 1, H/16, W/16] => [BD, 8, H/8, W/8]
        x2 = self.conv1(self.deconv_1(x1)) # [BD, 8, H/16, W/16] => [BD, 16, H/4, W/4]
        x3 = self.conv2(self.deconv_2(x2)) # [BD, 16, H/16, W/16] => [BD, 32, H/2, W/2]
        x4 = self.conv3(self.deconv_3(x3)) # [BD, 32, H/16, W/16] => [BD, 64, H, W]
        x4 = x4.permute(0, 2, 3, 1)
        out = self.linear2(x4) # [BD, 64, H, W] => [BD, 1, H, W]
        out = out.permute(0, 3, 1, 2)
        return out


class FeatNet(nn.Module):
    def __init__(self):
        super(FeatNet, self).__init__()
        base_filter = 8
        self.conv0_0 = convgnrelu(3, base_filter * 2, kernel_size=3, stride=1, dilation=1)
        self.conv0_1 = convgnrelu(base_filter * 2, base_filter * 4, kernel_size=3, stride=1, dilation=1)
        self.conv0_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=2)
        self.conv0_3 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

        # conv1_2 with conv0_2
        self.conv1_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=3)
        self.conv1_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

        # conv2_2 with conv0_2
        self.conv2_1 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=4)
        self.conv2_2 = convgnrelu(base_filter * 4, base_filter * 4, kernel_size=3, stride=1, dilation=1)

        # with concat conv0_3, conv1_2, conv2_2
        self.conv = nn.Conv2d(base_filter * 12, base_filter*4, 3, 1, 1)
 

    def forward(self, x):
        # x: [N, 3, 512, 512]
        conv0_0 = self.conv0_0(x) # [N, 16, 512, 512] 
        conv0_1 = self.conv0_1(conv0_0) # [N, 32, 512, 512] 
        conv0_2 = self.conv0_2(conv0_1) # [N, 32, 512, 512] 
        conv0_3 = self.conv0_3(conv0_2) # [N, 32, 512, 512] 

        conv1_2 = self.conv1_2(self.conv1_1(conv0_2)) #[N, 32, 512, 512] 

        conv2_2 = self.conv2_2(self.conv2_1(conv0_2)) #[N, 32, 512, 512] 

        conv = self.conv(torch.cat([conv0_3, conv1_2, conv2_2], 1)) #[N, 32, 512, 512]
        return conv

# input 3D Feature Volume
class UNetConvLSTM(nn.Module): # input 3D feature volume
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 bias=True, return_all_layers=False, gn=True):
        super(UNetConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size #feature: height, width)
        self.gn = gn
        print('Training Phase in UNetConvLSTM: {}, {}, gn: {}'.format(self.height, self.width, self.gn))
        self.input_dim  = input_dim # input channel
        self.hidden_dim = hidden_dim # output channel [16, 16, 16, 16, 16, 8]
        self.kernel_size = kernel_size # kernel size  [[3, 3]*5]
        self.num_layers = num_layers # Unet layer size: must be odd
        self.bias = bias #
        self.return_all_layers = return_all_layers

        cell_list = []
        #assert self.num_layers % 2  == 1 # Even
        self.down_num = (self.num_layers+1) / 2 

        # use GN 
        for i in range(0, self.num_layers):
            #cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            scale = 2**i if i < self.down_num else 2**(self.num_layers-i-1)
            print("layer:{}, scale:{}".format(i, scale))
            cell_list.append(ConvGnLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
            #cell_list.append(ConvLSTMCell(input_size=(int(self.height/scale), int(self.width/scale)),
                                        input_dim=self.input_dim[i],
                                        hidden_dim=self.hidden_dim[i],
                                        kernel_size=self.kernel_size[i],
                                        bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        self.deconv_0 = deConvGnReLU(
            16,
            16, #16
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
            )
        self.deconv_1 = deConvGnReLU(
            16,
            16, #16
            kernel_size=3,
            stride=2,
            padding=1,
            bias=self.bias,
            output_padding=1
            )
        self.conv_0 = nn.Conv2d(8, 1, 3, 1, padding=1)

    def forward(self, input_tensor, hidden_state, idx):
        """
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if idx ==0 : # input the first layer of input image
            for i in range(self.num_layers):
                scale = 2**i if i < self.down_num else 2**(self.num_layers-i-1)
                self.cell_list[i].update_size(input_tensor.size(2)//scale, input_tensor.size(3)//scale)    
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        cur_layer_input = input_tensor

        h0, c0 = hidden_state[0]= self.cell_list[0](input_tensor=cur_layer_input,cur_state=hidden_state[0])
        h0_1 = nn.MaxPool2d((2, 2), stride=2)(h0)
        h1, c1 = hidden_state[1] = self.cell_list[1](input_tensor=h0_1, cur_state=hidden_state[1])

        h1_0 = nn.MaxPool2d((2, 2), stride=2)(h1)  
        h2, c2 = hidden_state[2] = self.cell_list[2](input_tensor=h1_0, cur_state=hidden_state[2])

        h2_0 = self.deconv_0(h2) # auto reuse
        h2_1 = torch.cat([h2_0, h1], 1)
        h3, c3 = hidden_state[3] = self.cell_list[3](input_tensor=h2_1, cur_state=hidden_state[3])

        h3_0 = self.deconv_1(h3) # auto reuse
        h3_1 = torch.cat([h3_0, h0], 1)
        h4, c4 = hidden_state[4] = self.cell_list[4](input_tensor=h3_1,cur_state=hidden_state[4])
            
        cost = self.conv_0(h4) # auto reuse
        return cost, hidden_state

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

