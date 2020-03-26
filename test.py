import argparse
import collections
import numpy as np
import torch
from torchvision import transforms
from retinanet import model
from retinanet.dataloader import CocoDataset, FLIRDataset, CSVDataset, collater, \
                                 Resizer, AspectRatioBasedSampler, Augmenter, \
                                 Normalizer
from torch.utils.data import DataLoader
from retinanet import coco_eval
from retinanet.flir_eval import evaluate_flir
from retinanet import csv_eval

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'
# LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
# DATE_FORMAT = '%Y%m%d %H:%M:%S'

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def test(dataset, model, epoch, args, logger=None):
    # print("{} epoch: \t start validation....".format(epoch))
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
                # log_file.write(summarize)
                # log_file.write('\n')
            
        else:
            print('ERROR: Unknow dataset.')

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple testing script.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--dataset_root',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--resume', type=str,
                        help='Checkpoint state_dict file to resume training from')

    print(args)
    parser = parser.parse_args(args)

    # print('network_name:', network_name)
    testing_logger = logging.getLogger('Testing Logger')
    formatter      = logging.Formatter(LOGGING_FORMAT)
    streamhandler  = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    testing_logger.addHandler(streamhandler)

    # Create the data loaders
    if parser.dataset == 'coco':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on COCO,')
        print('Not support now.')
        assert 0
    elif parser.dataset == 'FLIR':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 1.2
        dataset_valid = FLIRDataset(parser.dataset_root, set_name='val',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Resizer(min_side=int(512*_scale), max_side=int(640*_scale))]))
    elif parser.dataset == 'thermal':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 1.2
        dataset_valid = FLIRDataset(parser.dataset_root, set_name='valid',
                                    transform=transforms.Compose(
                                        [
                                           Normalizer(), 
                                           Resizer(min_side=int(512*_scale), max_side=int(640*_scale))]))
    else:
        raise ValueError('Dataset type not understood (must be FLIR, COCO or csv), exiting.')
    
    build_param = {'logger': testing_logger}
    if parser.resume is not None:
        testing_logger.info('Loading Checkpoint : {}'.format(parser.resume))
        model = torch.load(parser.resume)
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
    #test(dataset_valid, retinanet, epoch_num, parser, testing_logger)
    # model = model.module
    model.eval()
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
            summarize = evaluate_flir(dataset_valid, model, valid_class_id=[0,])
            testing_logger.info('\n{}'.format(summarize))
            
        else:
            print('ERROR: Unknow dataset.')


if __name__ == '__main__':
    main()
