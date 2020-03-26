import argparse
import collections
import numpy as np
import torch
import torch.optim as optim
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
        elif args.dataset == 'thermal':
            summarize = evaluate_flir(dataset, model)
            if logger:
                logger.info('\n{}'.format(summarize))
            
        else:
            print('ERROR: Unknow dataset.')

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    # parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--dataset_root',
                        default='/root/data/VOCdevkit/',
                        help='Dataset root directory path [/root/data/VOCdevkit/, /root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
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

    parser = parser.parse_args(args)

    network_name = 'RetinaNet-Res{}'.format(parser.depth)
    # print('network_name:', network_name)
    net_logger    = logging.getLogger('Network Logger')
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    net_logger.addHandler(streamhandler)
    if parser.log:
        net_logger.setLevel(logging.INFO)
        # logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, 
        #                     filename=os.path.join('log', '{}.log'.format(network_name)), filemode='a')
        filehandler = logging.FileHandler(os.path.join('log', '{}_{}.log'.format(parser.dataset, network_name)), mode='a')
        filehandler.setFormatter(formatter)
        net_logger.addHandler(filehandler)

    net_logger.info('Network Name: {:>20}'.format(network_name))

    # Create the data loaders
    if parser.dataset == 'coco':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on COCO,')
        dataset_train = CocoDataset(parser.dataset_root, set_name='train2017',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Augmenter(), 
                                            Resizer()]))
        dataset_valid = CocoDataset(parser.dataset_root, set_name='val2017',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer()]))
    elif parser.dataset == 'FLIR':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on FLIR,')
        _scale = 1.2
        dataset_train = FLIRDataset(parser.dataset_root, set_name='train',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Augmenter(), 
                                            Resizer(min_side=int(512*_scale), max_side=int(640*_scale), logger=net_logger)]))
        dataset_valid = FLIRDataset(parser.dataset_root, set_name='val',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer(min_side=int(512*_scale), max_side=int(640*_scale))]))
    elif parser.dataset == 'thermal':
        if parser.dataset_root is None:
            raise ValueError('Must provide --dataset_root when training on thermal,')
        _scale = 0.2
        _w = 400 * _scale
        _h = 300 * _scale
        dataset_train = FLIRDataset(parser.dataset_root, set_name='train',
                                    transform=transforms.Compose(
                                        [
                                            Normalizer(), 
                                            Augmenter(), 
                                            Resizer(min_side=int(_h), max_side=int(_w), logger=net_logger)]))
        dataset_valid = FLIRDataset(parser.dataset_root, set_name='valid',
                                  transform=transforms.Compose(
                                      [
                                          Normalizer(), 
                                          Resizer(min_side=int(_h), max_side=int(_w))]))
    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        if parser.csv_val is None:
            dataset_valid = None
            print('No validation annotations provided.')
        else:
            dataset_valid = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be FLIR, COCO or csv), exiting.')
    
    

    # Original RetinaNet code
    # sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    # dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)
    # if dataset_valid is not None:
    #     sampler_val = AspectRatioBasedSampler(dataset_valid, batch_size=1, drop_last=False)
    #     dataloader_val = DataLoader(dataset_valid, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    dataloader_train = DataLoader(dataset_train,
                              batch_size=parser.batch_size,
                              num_workers=parser.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    dataloader_val = DataLoader(dataset_valid,
                              batch_size=1,
                              num_workers=parser.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)
    
    build_param = {'logger': net_logger}
    if parser.resume is not None:
        net_logger.info('Loading Checkpoint : {}'.format(parser.resume))
        retinanet = torch.load(parser.resume)
        s_b = parser.resume.rindex('_')
        s_e = parser.resume.rindex('.')
        start_epoch = int(parser.resume[s_b+1:s_e]) + 1
        net_logger.info('Continue on {} Epoch'.format(start_epoch))
    else:
        # Create the model
        if parser.depth == 18:
            retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, **build_param)
        elif parser.depth == 34:
            retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, **build_param)
        elif parser.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, **build_param)
        elif parser.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, **build_param)
        elif parser.depth == 152:
            retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, **build_param)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
        start_epoch = 0

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    net_logger.info('Weight Decay  : {}'.format(parser.weight_decay))
    net_logger.info('Learning Rate : {}'.format(parser.lr))

    # optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    optimizer = optim.Adam(retinanet.parameters(), lr=parser.lr, weight_decay=parser.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    # print('Num training images: {}'.format(len(dataset_train)))
    net_logger.info('Num Training Images: {}'.format(len(dataset_train)))

    for epoch_num in range(start_epoch, parser.epochs):
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                print(data['img'][0,:,:,:].shape)
                exit(0)
                # print(data['annot'])
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
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
        if (epoch_num + 1) % 5 == 0:
            test(dataset_valid, retinanet, epoch_num, parser, net_logger)

        # if parser.dataset == 'coco':

        #     print('Evaluating dataset')

        #     coco_eval.evaluate_coco(dataset_valid, retinanet)

        # elif parser.dataset == 'csv' and parser.csv_val is not None:

        #     print('Evaluating dataset')

        #     mAP = csv_eval.evaluate(dataset_valid, retinanet)

        scheduler.step(np.mean(epoch_loss))
        print('Learning Rate:', str(scheduler._last_lr))
        torch.save(retinanet.module, os.path.join(
                   'saved', '{}_{}_{}.pt'.format(parser.dataset, network_name, epoch_num)))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')
    # log_file.close()


if __name__ == '__main__':
    main()
