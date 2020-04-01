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

assert torch.__version__.split('.')[0] == '1'

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

def test(dataset, model, epoch, args, logger=None):
    logger.info("{} epoch: \t start validation....".format(epoch))
    # model = model.module
    model.eval()
    model.is_training = False
    with torch.no_grad():
        if(args.dataset == 'VOC'):
            evaluate(dataset, model)
        elif args.dataset == 'COCO':
            evaluate_coco(dataset, model)
        elif args.dataset == 'FLIR':
            summarize = evaluate_flir(dataset, model)
            if logger:
                logger.info('\n{}'.format(summarize))
        elif args.dataset == 'thermal':
            summarize = coco_evaluate_cvidata(dataset, model)
            if logger:
                logger.info('\n{}'.format(summarize))
            
        else:
            print('ERROR: Unknow dataset.')

def main():
    parser = argparse.ArgumentParser(description='Simple testing script.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--dataset_root',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--resume', type=str,
                        help='Checkpoint state_dict file to resume training from')

    args = parser.parse_args()

    # print('network_name:', network_name)
    testing_logger = logging.getLogger('Testing Logger')
    formatter      = logging.Formatter(LOGGING_FORMAT)
    streamhandler  = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    testing_logger.addHandler(streamhandler)

    # Create the data loaders
    if args.dataset == 'coco':
        if args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on COCO,')
        print('Not support now.')
        assert 0
    elif args.dataset == 'FLIR':
        if args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 1.2
        dataset_valid = FLIRDataset(args.dataset_root, set_name='val',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Resizer(min_side=int(512*_scale), max_side=int(640*_scale))]))
    elif args.dataset == 'thermal':
        if args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 0.2
        _w = 400 * _scale
        _h = 300 * _scale
        dataset_valid = CVIDataset(args.dataset_root, set_name='valid', annotation_name='thermal_annotations.json',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer(min_side=int(_h), max_side=int(_w))]))
    else:
        raise ValueError('Dataset type not understood (must be FLIR, COCO or csv), exiting.')
    
    build_param = {'logger': testing_logger}
    if args.resume is not None:
        testing_logger.info('Loading Checkpoint : {}'.format(args.resume))
        model = torch.load(args.resume)
    else:
        raise ValueError('Must provide --resume when testing.')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    testing_logger.info('There are {} images for testing.'.format(len(dataset_valid)))
    testing_logger.info('Testing start ...')
    model.is_training = False
    with torch.no_grad():
        if(parser.dataset == 'VOC'):
            evaluate(dataset, model)
        elif parser.dataset == 'COCO':
            evaluate_coco(dataset, model)
        elif parser.dataset == 'FLIR':
            summarize = evaluate_flir(dataset_valid, model)
            testing_logger.info('\n{}'.format(summarize))
        elif parser.dataset == 'thermal':
            summarize = coco_evaluate_cvidata(dataset_valid, model)
            testing_logger.info('\n{}'.format(summarize))
            
        else:
            print('ERROR: Unknow dataset.')



if __name__ == '__main__':
    main()
