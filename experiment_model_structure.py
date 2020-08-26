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
from torch.utils.data import DataLoader

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

KSEVEN_MODEL = 'ksevendet'

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

    support_architectures = [
        KSEVEN_MODEL,
    ]
    # support_architectures += [f'efficientdet-d{i}' for i in range(8)]
    # support_architectures += [f'retinanet-res{i}' for i in [18, 34, 50, 101, 152]]
    # print(support_architectures)

    if args.architecture == 'ksevendet':
        ksevendet_cfg = args.model_cfg
        if ksevendet_cfg.get('variant'):
            backbone_ = ksevendet_cfg['variant']
            neck_ = ksevendet_cfg['neck']
            head_ = ksevendet_cfg['head']
            network_name = f'{args.architecture}-{backbone_}-{neck_}-{head_}'
        else:
            assert isinstance(ksevendet_cfg, dict)
            network_name = f'{args.architecture}-{ksevendet_cfg["backbone"]}_specifical-{ksevendet_cfg["neck"]}'
    elif args.architecture in support_architectures:
        network_name = args.architecture
    else:
        raise ValueError('Architecture {} is not support.'.format(args.architecture))

    args.network_name = network_name
 
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
        filehandler = logging.FileHandler(os.path.join('log', 'experiment_pruning_structure_{}.log'.format(
                                                       args.network_name)), mode='a')
        filehandler.setFormatter(formatter)
        my_logger.addHandler(filehandler)

    return my_logger

def main():
    args = get_args()

    net_logger = get_logger(name='Network Logger', args=args)
    net_logger.info('Network Name: {}'.format(args.network_name))

    height, width = tuple(map(int, args.input_shape.split(',')))
    
    net_logger.info('Number of Classes: {:>3}'.format(args.num_classes))
    
    build_param = {'logger': net_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(args.model_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        net_model = net_model.cuda()

    # the following statement is unnecessary.
    net_model.set_onnx_convert_info(fixed_size=(height, width))    

    sample_image = np.zeros((height, width, 3)).astype(np.float32)
    sample_image = torch.from_numpy(sample_image)
    sample_input = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    sample_input_shape = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape


    net_pruner = KSevenPruner(net_model, input_shape=(height, width, 3), **build_param)
    DUMP_TENSOR_PRUNING_DEPENDENCY = True
    TENSOR_PRUNING_DEPENDENCY_JSON_PATH = 'tensor_pruning_info/{}_tensor_pruning_dependency.json'.format(args.network_name)
    tensor_pruning_dependency = net_pruner.gen_tensor_pruning_dependency(
                                    dump_json=DUMP_TENSOR_PRUNING_DEPENDENCY,
                                    dump_json_path=TENSOR_PRUNING_DEPENDENCY_JSON_PATH)
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
