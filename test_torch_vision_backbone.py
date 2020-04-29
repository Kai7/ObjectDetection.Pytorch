import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.models
from architectures import retinanet, efficientdet, ksevendet
from architectures.backbone import shufflenetv2, densenet, mnasnet, mobilenet
from datasettool.dataloader import KSevenDataset, CocoDataset, FLIRDataset, CVIDataset, CSVDataset, collater, \
                                 Resizer, AspectRatioBasedSampler, Augmenter, \
                                 Normalizer
from torch.utils.data import DataLoader
from datasettool import coco_eval
from datasettool.flir_eval import evaluate_flir
from datasettool.coco_eval import evaluate_coco
from datasettool.cvidata_eval import coco_evaluate_cvidata
from datasettool.ksevendata_eval import coco_evaluate_ksevendata
from datasettool import csv_eval

from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights


import os
import sys
import logging

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


in_h, in_w = 512, 512

dummy_input = torch.randn(1, 3, in_h, in_w, device='cuda')

BACKBONE = 'shuflenetv2'
BACKBONE = 'densenet'
# BACKBONE = 'mnasnet'
BACKBONE = 'mobilenet'

ksevendet_cfg = {
    'backbone'      : BACKBONE,
    'head'          : 'fpn',
    'num_classes'   : 2,
    'feature_pyramid_levels' : [3, 4, 5],
    'head_pyramid_levels'    : [3, 4, 5, 6, 7],
}
# net_model = shufflenetv2.shufflenet_v2_x0_5()
# net_model = densenet.densenet121()
# net_model = mnasnet.mnasnet0_5()
# net_model = mobilenet.mobilenet_v2()

net_model = ksevendet.KSevenDet(ksevendet_cfg)

print(net_model)

if torch.cuda.is_available():
    net_model = net_model.cuda()

if isinstance(net_model, ksevendet.KSevenDet):
    out = net_model(dummy_input, return_head=True)
else:
    out = net_model(dummy_input)

print('Done')