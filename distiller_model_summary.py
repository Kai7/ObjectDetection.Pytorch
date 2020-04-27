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
from datasettool import coco_eval
from datasettool.flir_eval import evaluate_flir
from datasettool import csv_eval

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

from distiller import model_summaries

assert torch.__version__.split('.')[0] == '1'

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_args():
    parser = argparse.ArgumentParser(description='Simple testing script.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--dataset_root',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--architecture', default='RetinaNet', type=str,
                        help='Network Architecture.')
    parser.add_argument('--num_classes', type=int,
                        help='The number of class.')
    parser.add_argument('--resume', type=str,
                        help='Checkpoint state_dict file to resume training from')

    return parser.parse_args()


def get_logger(description='Logger'):
    logger = logging.getLogger('Distiller Summary Logger')
    formatter      = logging.Formatter(LOGGING_FORMAT)
    streamhandler  = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    logger.addHandler(streamhandler)

    return logger


def main():
    args = get_args()
    if args.dataset == 'thermal':
        input_height, input_width = 60, 80
    elif args.dataset == '3s-pocket-thermal-face':
        input_height, input_width = 288, 384
        dataset_valid = CVIDataset(args.dataset_root, set_name='train', annotation_name='annotations.json',
                                   transform=transforms.Compose([    
                                                 Normalizer(), 
                                                 Resizer(height=input_height, width=input_width)]))
    else:
        raise ValueError('unknow dataset.')
    transform = transforms.Compose([Normalizer(inference_mode=True), 
                                    Resizer(height=input_height, width=input_width, inference_mode=True)])

    # print('network_name:', network_name)
    my_logger = get_logger(description='Distiller Summary Logger')

    my_logger.info('Build pytorch model...')
    build_param = {'logger': my_logger}
    if args.architecture == 'RetinaNet':
        model = retinanet.retinanet(args.depth, num_classes=args.num_classes, **build_param)
    elif args.architecture == 'RetinaNet-Tiny':
        model = retinanet.retinanet_tiny(num_classes=args.num_classes, **build_param)
    elif args.architecture == 'RetinaNet_P45P6':
        model = retinanet.retinanet_p45p6(num_classes=args.num_classes, **build_param)
    else:
        raise ValueError('Architecture <{}> unknown.'.format(args.architecture))
    
    my_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
    model.load_state_dict(torch.load(args.resume))

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # pdb.set_trace()

    #sample_input_shape = dataset_valid[0]['img'].shape
    sample_input_shape = dataset_valid[0]['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape

    print(sample_input_shape)

    model.eval()
    model_summaries.model_summary(model, 'compute', input_shape=sample_input_shape)



if __name__ == '__main__':
    main()
