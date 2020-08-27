import argparse
import yaml
import json
import collections
import numpy as np
import torch
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
import ksevendet.architecture.backbone.registry as registry
from ksevendet.ksevenpruner import KSevenPruner
from torch.utils.data import DataLoader

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

assert torch.__version__.split('.')[0] == '1'

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

KSEVEN_MODEL = 'ksevendet'

def get_args():
    parser = argparse.ArgumentParser(description='KSeven Pytorch to ONNX tool.')
    parser.add_argument('--model_config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--architecture', default='ksevendet', type=str,
                        help='Network Architecture.')
    parser.add_argument('--backbone', default='resnet', type=str,
                        help='KSevenDet backbone.')
    parser.add_argument('--neck', default='fpn', type=str,
                        help='KSevenDet neck.')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--tensor_dependency', default=None, type=str,
                        help='Tensor pruning dependency json file path.')
    parser.add_argument('--pruning_rate', default=0.0, type=float,
                        help='Uniform pruning rate.')
    parser.add_argument('--pruning_config', default=None, type=str,
                        help='Special pruning config path.')
    parser.add_argument('--onnx_name', type=str,
                        help='ONNX name')
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images (height, width)')
    parser.add_argument('--num_classes', type=int,
                        help='The number of class.')

    args = parser.parse_args()
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_cfg = yaml.safe_load(f)
        setattr(args, 'architecture', model_cfg.pop('architecture'))
        setattr(args, 'model_cfg', model_cfg)
    support_architectures = [
        KSEVEN_MODEL,
    ]
    print("Support Architectures:")
    print(support_architectures)

    if args.architecture == KSEVEN_MODEL:
        model_cfg = args.model_cfg
        if model_cfg.get('variant'):
            backbone_ = model_cfg['variant']
            neck_ = model_cfg['neck']
            head_ = model_cfg['head']
            network_name = f'{args.architecture}-{backbone_}-{neck_}-{head_}'
        else:
            assert isinstance(model_cfg, dict)
            network_name = f'{args.architecture}-{args.model_cfg["backbone"]}_specifical-{args.model_cfg["neck"]}'
    else:
        raise ValueError('Architecture {} is not support.'.format(args.architecture))
    args.network_name = network_name

    tensor_pruning_dependency = None
    eq_tensors_ids = None
    if args.tensor_dependency is not None:
        try:
            tensor_pruning_dependency = json.load(open(args.tensor_dependency))
            eq_tensors_ids = list(tensor_pruning_dependency.keys())
            eq_tensors_ids.sort()
        except FileNotFoundError:
            print('[WARNNING] Tensor Dependency File not found.')

    if args.pruning_rate > 0:
        assert tensor_pruning_dependency is not None, 'tensor_pruning_dependency must be not None.'
        pruning_tensor_cfg = list()
        for eq_t_id in eq_tensors_ids:
            pruning_args = {
                'pruning_type': 'random',
                'pruning_rate': args.pruning_rate,
            }
            pruning_tensor_cfg.append([eq_t_id, pruning_args])
        cfg_name='px{}_uniform'.format(100-int(args.pruning_rate * 100))
    elif args.pruning_config is not None:
        # TODO: Support pruning config
        assert 0, 'not support now'
        pruning_tensor_cfg = list() 
        cfg_name=''
    else:
        pruning_tensor_cfg = None
        cfg_name=''
    args.tensor_pruning_dependency = tensor_pruning_dependency
    args.pruning_tensor_cfg = pruning_tensor_cfg
    args.cfg_name = cfg_name

    height, width = _shape_1, _shape_2 = tuple(map(int, args.input_shape.split(',')))
    args.height = height
    args.width  = width

    return args


def main():
    args = get_args()
    my_logger = get_logger(name='KSeven Pytorch2ONNX Logger', args=args)
    my_logger.info('Network Name : {}'.format(args.network_name))
    my_logger.info('Name         : {}'.format(args.onnx_name))
    my_logger.info('Classes Num  : {}'.format(args.num_classes))
    my_logger.info(f'Input Tensor Size: [ {args.height}, {args.width}]')

    build_param = {'logger': my_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(args.model_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
    # elif args.architecture.split('-')[0] == 'retinanet':
    #     net_model = retinanet.build_retinanet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
    # elif args.architecture.split('-')[0] == 'efficientdet':
    #     net_model = efficientdet.build_efficientdet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        net_model = net_model.cuda()

    net_pruner = KSevenPruner(net_model, input_shape=(args.height, args.width, 3), 
                              tensor_pruning_dependency=args.tensor_pruning_dependency, **build_param)

    if args.pruning_tensor_cfg is not None:
        assert args.cfg_name, 'cfg_name must be given.'
        assert args.tensor_pruning_dependency, 'tensor_pruning_dependency must be given.'

        # if args.tensor_pruning_dependency is None:
        #     DUMP_TENSOR_PRUNING_DEPENDENCY = True
        #     TENSOR_PRUNING_DEPENDENCY_JSON_PATH = \
        #         os.path.join('model_pruning_config', '{}_tensor_pruning_dependency.json'.format(args.network_name))
        #     args.tensor_pruning_dependency = net_pruner.gen_tensor_pruning_dependency(
        #                                          dump_json=DUMP_TENSOR_PRUNING_DEPENDENCY,
        #                                          dump_json_path=TENSOR_PRUNING_DEPENDENCY_JSON_PATH)
        eq_tensors_ids = list(args.tensor_pruning_dependency.keys())
        eq_tensors_ids.sort()

        my_logger.info('Start Pruning.')
        net_pruner.prune(args.pruning_tensor_cfg)
        my_logger.info('Pruning Complete.')

    if args.resume is not None:
        #my_logger.info('Loading Checkpoint : {}'.format(args.resume))
        #model.load_state_dict(torch.load(args.resume))
        my_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
        try:
            ret = net_model.load_state_dict(torch.load(args.resume), strict=False)
        except RuntimeError as e:
            my_logger.warning(f'Ignoring {e}')
            my_logger.warning('Don\'t panic if you see this,')
            my_logger.warning('  this might be because you load a pretrained weights with different number of classes.')
            my_logger.warning('The rest of the weights should be loaded already.')

        net_model.set_onnx_convert_info(fixed_size=(args.height, args.width))
    else:
        raise ValueError('Must provide --resume when testing.')

    net_model.eval()

    dummy_input = torch.randn(1, 3, args.height, args.width, device='cuda')
    # in_tensor = torch.ones([1, 3, h, w])

    # torch.onnx.export(model, dummy_input, "{}.onnx".format(name), verbose=True, input_names=input_names, output_names=output_names)
    #torch.onnx.export(model.module, dummy_input, '{}.onnx'.format(name), verbose=True)
    # torch.onnx.export(model.module, dummy_input, './{}.onnx'.format(name))
    # torch.onnx.export(model.module, dummy_input, 'out.onnx', verbose=True, 
    #                   input_names=['input'], output_names=['scores', 'labels', 'boxes'])

    # ksevendet::forward(self, img_batch, annotations=None, return_head=False, return_loss=True)
    print('start export...')
    # model_input = [dummy_input, None, False, False]
    torch.onnx.export(net_model, dummy_input, '{}.onnx'.format(args.onnx_name), verbose=True, 
                      input_names=['input'], output_names=['classification', 'regression'],
                      opset_version=10,
                      do_constant_folding=True)
    # torch.onnx.export(net_model, dummy_input, '{}.onnx'.format(args.onnx_name), verbose=True, 
    #                   input_names=['input'], output_names=['classification', 'regression'], 
    #                   keep_initializers_as_inputs=True)
    print('export done.')

    print('Write to {}.onnx'.format(args.onnx_name))
    print('Done')


def get_logger(name='My Logger', args=None):
    LOGGING_FORMAT = '%(levelname)s:    %(message)s'
    # LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    # DATE_FORMAT = '%Y%m%d %H:%M:%S'

    my_logger     = logging.getLogger(name)
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    my_logger.addHandler(streamhandler)
    my_logger.setLevel(logging.INFO)

    return my_logger


if __name__ == '__main__':
    main()

    print('\ndone')
