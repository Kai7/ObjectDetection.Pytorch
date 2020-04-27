import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from architectures import retinanet
from datasettool.dataloader import CocoDataset, FLIRDataset, CVIDataset, CSVDataset, collater, \
                                 Resizer, AspectRatioBasedSampler, Augmenter, \
                                 Normalizer
from torch.utils.data import DataLoader
from datasettool import coco_eval
from datasettool.flir_eval import evaluate_flir
from datasettool.cvidata_eval import coco_evaluate_cvidata
from datasettool import csv_eval
import torchvision.models

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


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
        
    # print('network_name:', network_name)
    net_logger    = logging.getLogger('Network Logger')
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    net_logger.addHandler(streamhandler)
    net_logger.setLevel(logging.INFO)
    if args.log:
        filehandler = logging.FileHandler(os.path.join('log', '{}_{}.log'.format(args.dataset, network_name)), mode='a')
        filehandler.setFormatter(formatter)
        net_logger.addHandler(filehandler)

    net_logger.info('Network Name: {:>20}'.format(network_name))

    # Create the data loaders
    if args.dataset == 'coco':
        if args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on COCO,')
        dataset_train = CocoDataset(args.dataset_root, set_name='train2017',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Augmenter(), 
                                            Resizer()]))
        dataset_valid = CocoDataset(args.dataset_root, set_name='val2017',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer()]))
    elif args.dataset == 'FLIR':
        if args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 1.2
        dataset_train = FLIRDataset(args.dataset_root, set_name='train',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Augmenter(), 
                                            Resizer(min_side=int(512*_scale), max_side=int(640*_scale), logger=net_logger)]))
        dataset_valid = FLIRDataset(args.dataset_root, set_name='val',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer(min_side=int(512*_scale), max_side=int(640*_scale))]))
    elif args.dataset == 'thermal':
        if args.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on thermal,')
        _scale = 0.2
        #_scale = 1.5
        _w = 400 * _scale
        _h = 300 * _scale
        dataset_train = CVIDataset(args.dataset_root, set_name='train', annotation_name='thermal_annotations.json',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Augmenter(), 
                                            Resizer(min_side=int(_h), max_side=int(_w), logger=net_logger)]))
        dataset_valid = CVIDataset(args.dataset_root, set_name='valid', annotation_name='thermal_annotations.json',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer(min_side=int(_h), max_side=int(_w))]))
    else:
        raise ValueError('Dataset type not understood (must be FLIR, COCO or csv), exiting.')
    
    dataloader_train = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    dataloader_valid = DataLoader(dataset_valid,
                              batch_size=1,
                              num_workers=args.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)
    
    print('Number of Class: {:>3}'.format(dataset_train.num_classes()))
    # pdb.set_trace()
    
    build_param = {'logger': net_logger}
    if args.resume is not None:
        net_logger.info('Loading Checkpoint : {}'.format(args.resume))
        net_model = torch.load(args.resume)
        s_b = args.resume.rindex('_')
        s_e = args.resume.rindex('.')
        start_epoch = int(args.resume[s_b+1:s_e]) + 1
        net_logger.info('Continue on {} Epoch'.format(start_epoch))
    else:
        if args.architecture == 'RetinaNet':
            net_model = retinanet.retinanet(args.depth, num_classes=dataset_train.num_classes(), pretrained=True, **build_param)
        elif args.architecture == 'RetinaNet-Tiny':
            net_model = retinanet.retinanet_tiny(num_classes=dataset_train.num_classes(), **build_param)
        start_epoch = 0

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()

    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model).cuda()
    else:
        net_model = torch.nn.DataParallel(net_model)

    net_model.training = True

    net_logger.info('Weight Decay  : {}'.format(args.weight_decay))
    net_logger.info('Learning Rate : {}'.format(args.lr))

    # optimizer = optim.Adam(net_model.parameters(), lr=1e-5)
    optimizer = optim.Adam(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    net_model.train()
    net_model.module.freeze_bn()

    net_logger.info('Num Training Images: {}'.format(len(dataset_train)))

    for epoch_num in range(start_epoch, args.epochs):
        net_model.train()
        net_model.module.freeze_bn()

        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                # print(data['img'][0,:,:,:].shape)
                # exit(0)
                # print(data['annot'])
                if torch.cuda.is_available():
                    classification_loss, regression_loss = net_model([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = net_model([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), 0.1)
                optimizer.step()

                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                # if(iter_num % 10 == 0):
                if(iter_num % 100 == 0):
                    _log = 'Epoch: {:>3} | Iter: {:>4} | Class loss: {:1.5f} | BBox loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))
                    net_logger.info(_log)

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue


        # if (epoch_num + 1) % 1 == 0:
        # if (epoch_num + 1) % 5 == 0:
            # test(dataset_valid, net_model, epoch_num, args, net_logger)


        scheduler.step(np.mean(epoch_loss))
        print('Learning Rate:', str(scheduler._last_lr))
        torch.save(net_model.module, os.path.join(
                   'saved', '{}_{}_{}.pt'.format(args.dataset, network_name, epoch_num)))

    net_logger.info('Training Complete.')

    net_model.eval()
    test(dataset_valid, net_model, epoch_num, args, net_logger)

    torch.save(net_model.module, 'model_final.pt')


if __name__ == '__main__':
#    main()
    print('Done')