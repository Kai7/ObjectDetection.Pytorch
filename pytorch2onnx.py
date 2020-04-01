import argparse
import collections
import numpy as np
import torch
from torchvision import transforms
from architectures import retinanet
from datasettool.dataloader import CocoDataset, FLIRDataset, CVIDataset,CSVDataset, collater, \
                                   Resizer, AspectRatioBasedSampler, Augmenter, \
                                   Normalizer
from torch.utils.data import DataLoader

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

assert torch.__version__.split('.')[0] == '1'

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description='Simple testing script.')

parser.add_argument('--architecture', type=str,
                    help='Network Architecture.')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--name', type=str,
                    help='Model name')
parser.add_argument('--in_size', type=str,
                    help='Input size')

def main():
    args = parser.parse_args()

    if args.architecture == 'RetinaNet':
        if args.depth not in [18, 34, 50, 101, 152]:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
        network_name = 'RetinaNet-Res{}'.format(args.depth)
    elif args.architecture == 'RetinaNet-Tiny':
        network_name = 'RetinaNet-Tiny'
    else:
        raise ValueError('Architecture {} is not support.'.format(args.architecture)) 

    name = network_name.lower()

    net_logger    = logging.getLogger('Demo Logger')
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    net_logger.addHandler(streamhandler)
    net_logger.setLevel(logging.INFO)

    net_logger.info('network_name: {}'.format(network_name))
    net_logger.info('name: {}'.format(name))

    build_param = {'logger': net_logger}
    if args.resume is not None:
        net_logger.info('Loading Checkpoint : {}'.format(args.resume))
        model = torch.load(args.resume)
        model.module.convert_onnx = True
        model.module.fixed_size = (64, 80)
        model.module.fpn.convert_onnx = True
        model.module.regressionModel.convert_onnx = True
        model.module.classificationModel.convert_onnx = True
    else:
        raise ValueError('Must provide --resume when testing.')

    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()
    # model = torch.nn.DataParallel(model)

    #resnet18=resnet.resnet18(True)
    #checkpoint = torch.load("/home/shining/Downloads/resnet18-5c106cde.pth")
    
    #resnet18.load_state_dict(checkpoint)
    model.eval()

    h, w = 64, 80
    dummy_input = torch.randn(1, 3, h, w, device='cuda')
    # in_tensor = torch.ones([1, 3, h, w])

    # torch.onnx.export(model, dummy_input, "{}.onnx".format(name), verbose=True, input_names=input_names, output_names=output_names)
    #torch.onnx.export(model.module, dummy_input, '{}.onnx'.format(name), verbose=True)
    # torch.onnx.export(model.module, dummy_input, './{}.onnx'.format(name))
    # torch.onnx.export(model.module, dummy_input, 'out.onnx', verbose=True, 
    #                   input_names=['input'], output_names=['scores', 'labels', 'boxes'])
    torch.onnx.export(model.module, dummy_input, '{}.onnx'.format(name), verbose=True, 
                      input_names=['input'], output_names=['classification', 'transformed_anchors'])

    print('Write to {}.onnx'.format(name))
    print('Done')



if __name__ == '__main__':
    main()
