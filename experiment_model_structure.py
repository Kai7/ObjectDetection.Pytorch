from functools import partial

import argparse
import yaml
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
from ksevendet.ksevenpruner import KSevenPruner
import ksevendet.architecture.backbone.registry as registry
#from datasettool.dataloader import KSevenDataset, CocoDataset, collater, \
#                                   Resizer, AspectRatioBasedSampler, Augmenter, \
#                                   Normalizer
from torch.utils.data import DataLoader
#from datasettool import coco_eval
#from datasettool.flir_eval import evaluate_flir
#from datasettool.coco_eval import evaluate_coco
#from datasettool.cvidata_eval import coco_evaluate_cvidata
#from datasettool.ksevendata_eval import coco_evaluate_ksevendata
#from datasettool import csv_eval

# from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weightso

from PIL import Image

import os
import sys
import logging
# from thop import profile

from distiller import model_summaries
from distiller import create_png
from distiller import model_find_module_name
from distiller import SummaryGraph

import copy

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_args():
    parser = argparse.ArgumentParser(description='.')
    
    parser.add_argument('--model_config', default=None, type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--architecture', default='ksevendet', type=str,
                        help='Network Architecture.')
    parser.add_argument('--num_classes', type=int,
                        help='The number of class.')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images Shape (height, width)')
    parser.add_argument("--log", default=False, action="store_true" , 
                        help="Write log file.")

    args = parser.parse_args()

    assert args.model_config, 'Model config must be provide.'
    with open(args.model_config, 'r') as f:
        model_cfg = yaml.safe_load(f)
    setattr(args, 'architecture', model_cfg.pop('architecture'))
    setattr(args, 'model_cfg', model_cfg)
        
    return args

def get_logger(name='My Logger', args=None):
    LOGGING_FORMAT = '%(levelname)s:    %(message)s'

    my_logger     = logging.getLogger(name)
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    my_logger.addHandler(streamhandler)
    my_logger.setLevel(logging.INFO)
    if args is not None and args.log:
        filehandler = logging.FileHandler(os.path.join('log', '{}_{}.log'.format(args.dataset, args.network_name)), mode='a')
        filehandler.setFormatter(formatter)
        my_logger.addHandler(filehandler)

    return my_logger

def main():
    args = get_args()
    support_architectures = [
        'ksevendet',
    ]
    support_architectures += [f'efficientdet-d{i}' for i in range(8)]
    support_architectures += [f'retinanet-res{i}' for i in [18, 34, 50, 101, 152]]
    support_architectures.append('retinanet-p45p6')

    print(support_architectures)

    if args.architecture == 'ksevendet':
        ksevendet_cfg = args.model_cfg
        if ksevendet_cfg.get('variant'):
            network_name = f'{args.architecture}-{ksevendet_cfg["variant"]}-{ksevendet_cfg["neck"]}'
        else:
            assert isinstance(ksevendet_cfg, dict)
            network_name = f'{args.architecture}-{ksevendet_cfg["backbone"]}_specifical-{ksevendet_cfg["neck"]}'
    elif args.architecture in support_architectures:
        network_name = args.architecture
    else:
        raise ValueError('Architecture {} is not support.'.format(args.architecture))

    args.network_name = network_name

    net_logger = get_logger(name='Network Logger', args=args)
    net_logger.info('Network Name: {}'.format(network_name))

    height, width = tuple(map(int, args.input_shape.split(',')))
    
    net_logger.info('Number of Classes: {:>3}'.format(args.num_classes))
    
    build_param = {'logger': net_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    # load last weights
    if args.resume is not None:
        net_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
        try:
            ret = net_model.load_state_dict(torch.load(args.resume), strict=False)
        except RuntimeError as e:
            net_logger.warning(f'Ignoring {e}')
            net_logger.warning(f'Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        s_b = args.resume.rindex('_')
        s_e = args.resume.rindex('.')
        start_epoch = int(args.resume[s_b+1:s_e]) + 1
        net_logger.info('Continue on {} Epoch'.format(start_epoch))
    else:
        start_epoch = 1
        
    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()

    # the following statement is unnecessary.
    net_model.set_onnx_convert_info(fixed_size=(height, width))    

    sample_image = np.zeros((height, width, 3)).astype(np.float32)
    sample_image = torch.from_numpy(sample_image)
    sample_input = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    sample_input_shape = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape


    net_pruner = KSevenPruner(net_model, input_shape=(height, width, 3), **build_param)
    TENSOR_PRUNING_DEPENDENCY_JSON_PATH = '{}_tensor_pruning_dependency.json'.format(args.network_name)
    tensor_pruning_dependency = net_pruner.gen_tensor_pruning_dependency()
    eq_tensors_ids = list(tensor_pruning_dependency.keys())
    eq_tensors_ids.sort()

    PRUNING_RATE = 0.5

    for eq_t_id in eq_tensors_ids:
        pruning_tensor_cfg = list()
        pruning_args = {
            'pruning_type': 'random',
            'pruning_rate': PRUNING_RATE,
        }
        pruning_tensor_cfg.append([eq_t_id, pruning_args])
    
        net_logger.info('Start Pruning.')
        net_pruner.prune(pruning_tensor_cfg)
        net_logger.info('Pruning Finish.')

if __name__ == '__main__':
    main()
