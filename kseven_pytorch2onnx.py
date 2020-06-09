import argparse
import yaml
import collections
import numpy as np
import torch
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
import ksevendet.architecture.backbone.registry as registry
from torch.utils.data import DataLoader

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

assert torch.__version__.split('.')[0] == '1'

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

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
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images (height, width)')
    parser.add_argument('--num_classes', type=int,
                        help='The number of class.')
    parser.add_argument('--onnx_name', type=str,
                        help='ONNX name')
    parser.add_argument("--log", default=False, action="store_true" , 
                        help="Write log file.")

    args = parser.parse_args()
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_cfg = yaml.safe_load(f)
        setattr(args, 'architecture', model_cfg.pop('architecture'))
        setattr(args, 'model_cfg', model_cfg)

    return args

#parser = argparse.ArgumentParser(description='Simple testing script.')
#parser.add_argument('--architecture', type=str,
#                    help='Network Architecture.')
#parser.add_argument('--resume', type=str,
#                    help='Checkpoint state_dict file to resume training from')
#parser.add_argument('--name', type=str,
#                    help='Model name')
#parser.add_argument('--num_classes', type=int,
#                    help='The number of class.')
#parser.add_argument('--in_size', type=str, default='480,720',
#                    help='Input size [height, width]')

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

    args.network_name = network_name.lower()

    net_logger = get_logger(name='KSeven Pytorch2ONNX Logger', args=args)
    net_logger.info('Network Name : {}'.format(network_name))
    net_logger.info('Name         : {}'.format(args.onnx_name))
    net_logger.info('Classes Num  : {}'.format(args.num_classes))

    in_h, in_w = args.input_shape.split(',')
    in_h, in_w = int(in_h), int(in_w)
    net_logger.info(f'Input Tensor Size: [ {in_h}, {in_w}]')

    build_param = {'logger': net_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
    elif args.architecture == 'retinanet-p45p6':
        net_model = retinanet.retinanet_p45p6(num_classes=args.num_classes, **build_param)
    elif args.architecture.split('-')[0] == 'retinanet':
        net_model = retinanet.build_retinanet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
    elif args.architecture.split('-')[0] == 'efficientdet':
        net_model = efficientdet.build_efficientdet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    if args.resume is not None:
        #net_logger.info('Loading Checkpoint : {}'.format(args.resume))
        #model.load_state_dict(torch.load(args.resume))
        net_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
        try:
            ret = net_model.load_state_dict(torch.load(args.resume), strict=False)
        except RuntimeError as e:
            net_logger.warning(f'Ignoring {e}')
            net_logger.warning(f'Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        net_model.set_onnx_convert_info(fixed_size=(in_h, in_w))
        #net_model.fixed_size = (in_h, in_w)
        #net_model.convert_onnx = True
        #net_model.neck.convert_onnx = True
        #net_model.regressor.convert_onnx = True
        #net_model.classifier.convert_onnx = True
    else:
        raise ValueError('Must provide --resume when testing.')

    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()
    # model = torch.nn.DataParallel(model)

    #resnet18=resnet.resnet18(True)
    #checkpoint = torch.load("/home/shining/Downloads/resnet18-5c106cde.pth")
    
    #resnet18.load_state_dict(checkpoint)
    net_model.eval()

    dummy_input = torch.randn(1, 3, in_h, in_w, device='cuda')
    # in_tensor = torch.ones([1, 3, h, w])

    # torch.onnx.export(model, dummy_input, "{}.onnx".format(name), verbose=True, input_names=input_names, output_names=output_names)
    #torch.onnx.export(model.module, dummy_input, '{}.onnx'.format(name), verbose=True)
    # torch.onnx.export(model.module, dummy_input, './{}.onnx'.format(name))
    # torch.onnx.export(model.module, dummy_input, 'out.onnx', verbose=True, 
    #                   input_names=['input'], output_names=['scores', 'labels', 'boxes'])

    # ksevendet::forward(self, img_batch, annotations=None, return_head=False, return_loss=True)
    print('start export...')
    model_input = [dummy_input, None,False, False]
    torch.onnx.export(net_model, dummy_input, '{}.onnx'.format(args.onnx_name), verbose=True, 
                      input_names=['input'], output_names=['classification', 'regression'], 
                      keep_initializers_as_inputs=True)
    # torch.onnx.export(net_model, dummy_input, '{}.onnx'.format(args.onnx_name), verbose=True, 
    #                   input_names=['input'], output_names=['classification', 'regression'], 
    #                   keep_initializers_as_inputs=True)
    print('export done.')

    print('Write to {}.onnx'.format(args.onnx_name))
    print('Done')



if __name__ == '__main__':
    main()
