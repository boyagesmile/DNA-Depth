import torch
import torch.nn as nn
import thop

from efficientnet_pytorch.utils import Conv2dDynamicSamePadding
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from efficientnet_pytorch.utils import MemoryEfficientSwish
from thop.vision.basic_hooks import count_convNd, zero_ops

import networks
from ptflops.flops_counter import get_model_complexity_info

from torchstat import stat

class model_name(nn.Module):
    def __init__(self, encoder, decoder):
        super(model_name, self).__init__()
        self.encoder = encoder
        self.depth_decoder = decoder
    def forward(self, x):
        feat = self.encoder(x)
        result = self.depth_decoder(feat)

        return(result)

def count_sigmoid(m, x, y):
    """
    Using this approximation for exponetial operation:  exp(x) = 1 + x + x^2/2! + .. + x^9/9!
    For sigmoid f(x) = 1/(1+exp(x)): there are totally 10 add ops, 9 division ops(2! are considered as constant).
    Since it is element-wise operation. The final ops is about(10+9)*num_elements.
    """
    x = x[0]

    nelements = x.numel()

    total_ops = 19 * nelements
    m.total_ops += torch.Tensor([int(total_ops)])


def count_swish(m, x, y):
    """
    swish = x*sigmoid(x). So the total ops is 20*num_elements. See definition of count_sigmoid.
    """
    x = x[0]

    nelements = x.numel()

    total_ops = 20 * nelements
    m.total_ops += torch.Tensor([int(total_ops)])

class Test_Enconder(nn.Module):
    def __init__(self):
        super(Test_Enconder, self).__init__()
        num_enc = [16, 24, 40, 112, 320, 1280]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_enc[0], kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_enc[0], out_channels=num_enc[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=num_enc[1], out_channels=num_enc[2], kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=num_enc[2], out_channels=num_enc[3], kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=num_enc[3], out_channels=num_enc[4], kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=num_enc[4], out_channels=num_enc[5], kernel_size=1, stride=1)
    def forward(self, x):
        feats = []
        feats.append(self.conv1(x))
        feats.append(self.conv2(feats[-1]))
        feats.append(self.conv3(feats[-1]))
        feats.append(self.conv4(feats[-1]))
        feats.append(self.conv5(feats[-1]))
        feats.append(self.conv6(feats[-1]))

        return feats




if __name__ == '__main__':
    custom_ops = {
            Conv2dDynamicSamePadding: count_convNd,
            Conv2dStaticSamePadding: count_convNd,
            MemoryEfficientSwish: zero_ops,
            # MemoryEfficientSwish: count_swish,
        }

    encoder = networks.EfficientEncoder()
    depth_decoder = networks.EfficientDecoder(encoder.num_ch_enc)
    model = model_name(encoder, depth_decoder)
    verbose = False
    x = torch.randn(1, 3, 224, 224)
    args = dict(inputs=(x,), custom_ops=custom_ops, verbose=True)
    macs, params = thop.profile(encoder, **args)
    print(f'EfficientNet encoder: {params * 1e-6:.2f} M, {macs * 1e-9:.2f}G FLOPS')

    x = torch.randn(1, 3, 192, 640)
    args = dict(inputs=(x,), custom_ops=custom_ops, verbose=True)
    macs, params = thop.profile(model, **args)
    print(f'EfficientNet: {params * 1e-6:.2f} M, {macs * 1e-9:.2f}G FLOPS')

    # flops, params = get_model_complexity_info(model, (3, 192, 640), as_strings=True, print_per_layer_stat=True)
    flops, params = get_model_complexity_info(model, (3, 192, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops:  ' + flops)
    print('Params: ' + params)

    # encoder_test = Test_Enconder()
    # model_test = model_name(encoder_test, depth_decoder)
    # stat(model_test, (3, 192, 640))

    print('down')
    # text_path = "test.txt"
    # with open(text_path, "a") as f:
    #     f.write(('{:.3f}MB, {:.2f}G FLOPS').format(params * 1e-6, macs * 1e-9))
    #     f.write("\r\n")
