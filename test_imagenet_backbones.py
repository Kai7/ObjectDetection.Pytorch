import argparse
import collections
import numpy as np
import torch

# from architectures import retinanet, efficientdet, ksevendet
# from architectures.backbone import shufflenetv2, densenet, mnasnet, mobilenet
# from ksevendet.architecture import ksevendet
# from architecture.backbone import shufflenetv2, mobilenetv2, mobilenetv3, efficientnet, resnet
from ksevendet.architecture import ksevendet
from ksevendet.architecture.backbone import shufflenetv2, mobilenetv2, mobilenetv3, efficientnet, resnet, densenet, res2net, senet, sknet


from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


import os
import logging

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


in_h, in_w = 512, 512

dummy_input = torch.randn(1, 3, in_h, in_w, device='cuda')
# dummy_input = torch.randn(4, 3, in_h, in_w, device='cuda')

BACKBONE = 'shuflenetv2'
# BACKBONE = 'densenet'
# BACKBONE = 'mnasnet'
# BACKBONE = 'mobilenet'

ksevendet_cfg = {
    'backbone'      : BACKBONE,
    'neck'          : 'fpn',
    'num_classes'   : 2,
    'feature_pyramid_levels' : [3, 4, 5],
    'head_pyramid_levels'    : [3, 4, 5, 6, 7],
}

# net_model = mobilenetv3.mobilenetv3_large_075(features_only=True, pretrained=True)
# net_model = mobilenetv3.mobilenetv3_large_100(features_only=True, pretrained=True)
# net_model = mobilenetv3.mobilenetv3_small_075(features_only=True)
# net_model = mobilenetv3.mobilenetv3_small_100(features_only=True)
# net_model = mobilenetv3.mobilenetv3_rw(features_only=True)

# net_model = res2net.res2net50_14w_8s(features_only=True)
# net_model = res2net.res2net50_26w_4s(features_only=True)
# net_model = res2net.res2net50_26w_6s(features_only=True)
# net_model = res2net.res2net50_26w_8s(features_only=True)
# net_model = res2net.res2net50_48w_2s(features_only=True)
# net_model = res2net.res2net101_26w_4s(features_only=True)
# net_model = resnet.resnet18(features_only=True)
# net_model = resnet.resnet26(features_only=True)
# net_model = resnet.resnet26d(features_only=True)
# net_model = resnet.resnet34(features_only=True)
# net_model = resnet.resnet50(features_only=True)
# net_model = resnet.resnet50d(features_only=True)
# net_model = resnet.resnet101(features_only=True)
# net_model = resnet.resnet152(features_only=True)

# net_model = senet.seresnet18(features_only=True)
# net_model = senet.seresnet34(features_only=True)
# net_model = senet.seresnet50(features_only=True)
# net_model = senet.seresnet101(features_only=True)
# net_model = senet.seresnet152(features_only=True)
# net_model = senet.senet154(features_only=True)

# net_model = sknet.skresnet18(features_only=True)
# net_model = sknet.skresnet34(features_only=True)
# net_model = sknet.skresnet50(features_only=True)
# net_model = sknet.skresnet50d(features_only=True)
# net_model = sknet.skresnext50_32x4d(features_only=True)


# net_model = shufflenetv2.shufflenet_v2_x0_5(features_only=True)
# net_model = shufflenetv2.shufflenet_v2_x1_0(features_only=True
# net_model = shufflenetv2.shufflenet_v2_x1_5(features_only=True)
# net_model = shufflenetv2.shufflenet_v2_x2_0(features_only=True)


# net_model = mobilenetv2.mobilenet_v2(features_only=True)

# net_model = efficientnet.mobilenetv2_100(features_only=True)
# net_model = efficientnet.mobilenetv2_110d(features_only=True)
# net_model = efficientnet.mobilenetv2_120d(features_only=True)
# net_model = efficientnet.mobilenetv2_140(features_only=True)

# net_model = densenet.densenet121(features_only=True)
# net_model = densenet.densenet161(features_only=True)
# net_model = densenet.densenet169(features_only=True)
# net_model = densenet.densenet201(features_only=True)

# net_model = mnasnet.mnasnet0_5()
# net_model = mnasnet.mnasnet0_75()
# net_model = mnasnet.mnasnet1_0()
# net_model = mnasnet.mnasnet1_3()

# net_model = ksevendet.KSevenDet(ksevendet_cfg)

# print(net_model)

if torch.cuda.is_available():
    net_model = net_model.cuda()

net_model.eval()

if isinstance(net_model, ksevendet.KSevenDet):
    out = net_model(dummy_input, return_head=True)
else:
    out = net_model(dummy_input)

if hasattr(net_model, 'features_num'):
    print(net_model.features_num)

print(net_model.feature_channels())

# pdb.set_trace()
print('Done')