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

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'
# LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
# DATE_FORMAT = '%Y%m%d %H:%M:%S'

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
parser.add_argument('--dataset_root',
                    default='/root/data/VOCdevkit/',
                    help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/, /root/data/FLIR_ADAS]')
parser.add_argument('--architecture', default='RetinaNet', type=str,
                    help='Network Architecture.')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--log", default=False, action="store_true" , 
                    help="Write log file.")


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
    else:
        raise ValueError('Dataset type not understood (must be FLIR, COCO or csv), exiting.')
    
    dataloader_train = DataLoader(dataset_train,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
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

        scheduler.step(np.mean(epoch_loss))
        print('Learning Rate:', str(scheduler._last_lr))
        #torch.save(net_model.module, os.path.join(
        #           'saved', '{}_{}_{}.pt'.format(args.dataset, network_name, epoch_num)))

    net_logger.info('Training Complete.')
    _save_path = os.path.join('saved', '{}_{}_{}_final.pt'.format(args.dataset, network_name, args.epochs))
    net_logger.info('Save Final *.pt File to {}'.format(_save_path))
    torch.save(net_model, _save_path)


if __name__ == '__main__':
    main()
