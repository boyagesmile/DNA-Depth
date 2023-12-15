# from efficientnet.model import EfficientNet
from .efficientnet import EfficientNet
from .utils import *
import numpy as np

def load_pretrained_weights(model, model_name, advprop=False):
    url_map_ = url_map_advprop if advprop else url_map
    state_dict = model_zoo.load_url(url_map_[model_name])
    model_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in state_dict.items() if k in model_dict})
    return model


class EfficientEncoder(nn.Module):
    def __init__(self, valid_models='efficientnet-b0', pretrained=True, height=192, width=640):
        super(EfficientEncoder, self).__init__()
        assert valid_models in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                                'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                                'efficientnet-b8',
                                # Support the construction of 'efficientnet-l2' without pretrained weights
                                'efficientnet-l2'], "Can only run with efficientnet-b0 ~ b8 and efficientnet-l2 model"
        if valid_models in ['efficientnet-b0', 'efficientnet-b1']:
            self.num_ch_enc = np.array([16, 24, 40, 112, 320, 1280])
        elif valid_models == 'efficientnet-b2':
            self.num_ch_enc = np.array([16, 24, 48, 120, 352, 1408])
        elif valid_models == 'efficientnet-b3':
            self.num_ch_enc = np.array([24, 32, 48, 136, 384, 1536])
        elif valid_models == 'efficientnet-b4':
            self.num_ch_enc = np.array([24, 32, 56, 160, 448, 1792])
        elif valid_models == 'efficientnet-b5':
            self.num_ch_enc = np.array([24, 40, 64, 176, 512, 2048])
        elif valid_models == 'efficientnet-b6':
            self.num_ch_enc = np.array([32, 40, 72, 200, 576, 2304])
        elif valid_models == 'efficientnet-b7':
            self.num_ch_enc = np.array([32, 48, 80, 224, 640, 2560])
        elif valid_models == 'efficientnet-b8':
            self.num_ch_enc = np.array([32, 56, 88, 248, 704, 2816])
        else:
            self.num_ch_enc = np.array([72, 104, 176, 480, 1376, 5504])

        model = EfficientNet.from_name(valid_models, image_size=[height, width], include_top=False)
        # del model._conv_head
        # del model._bn1
        self.model = model
        if pretrained:
            assert valid_models in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                                    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                                    'efficientnet-b8'], " only efficientnet-b0 ~ b8 have pretrained weights"
            self.model = load_pretrained_weights(self.model, valid_models)

    def forward(self, inputs):
        feats = []
        endpoints = self.model(inputs)

        feats.append(endpoints['reduction_1'])
        feats.append(endpoints['reduction_2'])
        feats.append(endpoints['reduction_3'])
        feats.append(endpoints['reduction_4'])
        feats.append(endpoints['reduction_5'])
        return feats



if __name__ == '__main__':

    inputs = torch.rand(1, 3, 192, 640).cuda()
    model = EfficientNet.from_name('efficientnet-b0', image_size=[192, 640], include_top=False).cuda()
    from ptflops.flops_counter import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 192, 640), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)
    # image_size = [192, 640],
    endpoints = model(inputs)
    print(endpoints['reduction_1'].shape)
    print(endpoints['reduction_2'].shape)
    print(endpoints['reduction_3'].shape)
    print(endpoints['reduction_4'].shape)
    print(endpoints['reduction_5'].shape)
    encoder = EfficientEncoder('efficientnet-b0').cuda()
    total = sum([param.nelement() for param in encoder.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))


    feats = encoder(inputs)

    print("down")