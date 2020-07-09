import argparse
import yaml
import json
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from architectures import retinanet, efficientdet
# from architectures import ksevendet
from ksevendet.architecture import ksevendet
from ksevendet.ksevenpruner import KSevenPruner
import ksevendet.architecture.backbone.registry as registry
# from architectures.backbone import shufflenetv2, densenet, mnasnet, mobilenet
from datasettool.dataloader import KSevenDataset, CocoDataset, FLIRDataset, collater, \
                                   Resizer, AspectRatioBasedSampler, Augmenter, \
                                   Normalizer
from torch.utils.data import DataLoader
from datasettool import coco_eval
from datasettool.flir_eval import evaluate_flir
from datasettool.coco_eval import evaluate_coco
from datasettool.cvidata_eval import coco_evaluate_cvidata
from datasettool.ksevendata_eval import coco_evaluate_ksevendata
from datasettool import csv_eval

from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights

import os
import sys
import logging

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a object detection network.')
    
    parser.add_argument('--dataset', help='Dataset Name')
    parser.add_argument('--dataset_root', default='/root/data/',
                        help='Dataset root directory path [/root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--dataset_type', default='kseven',
                        help='Dataset type, must be one of kseven, coco, flir')
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
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images (height, width)')
    parser.add_argument('--resize_mode', default=1, type=int,
                        help='The resize mode for Resizer')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-num_gpus', default=1, type=int,
                        help='number of GPUs (default: 1)')
    parser.add_argument("--log", default=False, action="store_true" , 
                        help="Write log file.")
    parser.add_argument('--valid_period', default=5, type=int,
                        help='Batch size for training')
    parser.add_argument("--validation_only", default=False, action="store_true" , 
                        help="Only run validation.")

    args = parser.parse_args()
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_cfg = yaml.safe_load(f)
        setattr(args, 'architecture', model_cfg.pop('architecture'))
        setattr(args, 'model_cfg', model_cfg)
    # print(model_cfg)
    # pdb.set_trace()
        

    return args

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
        filehandler = logging.FileHandler(os.path.join('log', 'experiment', '{}_{}_{}.log'.format(
                                          args.dataset, args.network_name, args.cfg_name)), mode='a')
        filehandler.setFormatter(formatter)
        my_logger.addHandler(filehandler)

    return my_logger


def test(dataset, model, epoch, args, logger=None):
    logger.info("{} epoch: \t start validation....".format(epoch))
    # model = model.module
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, CustomDataParallel):
        model = model.module
    model.eval()
    model.is_training = False
    with torch.no_grad():
        if args.dataset_type == 'coco':
            evaluate_coco(dataset, model)
        elif args.dataset_type == 'flir':
            summarize = evaluate_flir(dataset, model)
            if logger:
                logger.info('\n{}'.format(summarize))
        elif args.dataset_type == 'kseven':
            summarize = coco_evaluate_ksevendata(dataset, model)
            if logger:
                logger.info('\n{}'.format(summarize))
        else:
            print('ERROR: Unknow dataset.')


def train_pruning_model(pruning_tensor_cfg, cfg_name='', tensor_pruning_dependency=None):
    assert tensor_pruning_dependency is not None, 'tensor_pruning_dependency is None'

    args = get_args()
    args.cfg_name = cfg_name

    assert args.dataset, 'dataset must provide'
    default_support_backbones = registry._module_to_models

    # write_support_backbones(default_support_backbones)

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

    args.network_name = network_name

    net_logger = get_logger(name='pruning_{}_{}'.format(
                            args.network_name, cfg_name), args=args)
    net_logger.info('Network Name: {}'.format(network_name))
    net_logger.info('Dataset Name: {}'.format(args.dataset))
    net_logger.info('Dataset Root: {}'.format(args.dataset_root))
    net_logger.info('Dataset Type: {}'.format(args.dataset_type))
    net_logger.info('Training Epochs : {}'.format(args.epochs))
    net_logger.info('Batch Size      : {}'.format(args.batch_size))
    net_logger.info('Weight Decay    : {}'.format(args.weight_decay))
    net_logger.info('Learning Rate   : {}'.format(args.lr))

    height, width = _shape_1, _shape_2 = tuple(map(int, args.input_shape.split(',')))
    _normalizer = Normalizer()
    #_augmenter  = Augmenter(scale_min=0.9, logger=net_logger)
    _augmenter  = Augmenter(use_scale=False, scale_min=0.9, logger=net_logger)
    if args.resize_mode == 0:
        _resizer = Resizer(min_side=_shape_1, max_side=_shape_2, resize_mode=args.resize_mode, logger=net_logger)
    elif args.resize_mode == 1:
        _resizer = Resizer(height=height, width=width, resize_mode=args.resize_mode, logger=net_logger)
    else:
        raise ValueError('Illegal resize mode.')
    transfrom_funcs_train = [
        _augmenter,
        _normalizer,
        _resizer,
    ]
    transfrom_funcs_valid = [
        _normalizer,
        _resizer,
    ]
    # Create the data loaders
    if args.dataset_type == 'kseven':
        dataset_train = KSevenDataset(args.dataset_root, set_name='train', transform=transforms.Compose(transfrom_funcs_train))
        # dataset_valid = KSevenDataset(args.dataset_root, set_name='train', transform=transforms.Compose(transfrom_funcs_valid))
        dataset_valid = KSevenDataset(args.dataset_root, set_name='valid', transform=transforms.Compose(transfrom_funcs_valid))
    elif args.dataset_type == 'coco':
        dataset_train = CocoDataset(args.dataset_root, set_name='train', transform=transforms.Compose(transfrom_funcs_train))
        dataset_valid = CocoDataset(args.dataset_root, set_name='valid', transform=transforms.Compose(transfrom_funcs_valid))
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
    
    net_logger.info('Number of Classes: {:>3}'.format(dataset_train.num_classes()))
    
    build_param = {'logger': net_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=dataset_train.num_classes(), pretrained=False, **build_param)
    elif args.architecture == 'retinanet-p45p6':
        net_model = retinanet.retinanet_p45p6(num_classes=dataset_train.num_classes(), **build_param)
    elif args.architecture.split('-')[0] == 'retinanet':
        net_model = retinanet.build_retinanet(args.architecture, num_classes=dataset_train.num_classes(), pretrained=False, **build_param)
    elif args.architecture.split('-')[0] == 'efficientdet':
        net_model = efficientdet.build_efficientdet(args.architecture, num_classes=dataset_train.num_classes(), pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    # load last weights
    if args.resume is not None:
        net_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
        try:
            ret = net_model.load_state_dict(torch.load(args.resume), strict=False)
        except RuntimeError as e:
            net_logger.warning(f'Ignoring {e}')
            net_logger.warning(f'Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        s_b = args.resume.rindex('_')
        s_e = args.resume.rindex('.')
        start_epoch = int(args.resume[s_b+1:s_e]) + 1
        net_logger.info('Continue on {} Epoch'.format(start_epoch))
    else:
        start_epoch = 1
        
    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()

    sample_image = np.zeros((height, width, 3)).astype(np.float32)
    sample_image = torch.from_numpy(sample_image)
    sample_input = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    sample_input_shape = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape

    # the following statement is unnecessary.
    # net_model.set_onnx_convert_info(fixed_size=(height, width))

    net_pruner = KSevenPruner(net_model, input_shape=(height, width, 3), 
                              tensor_pruning_dependency=tensor_pruning_dependency, **build_param)
    eq_tensors_ids = list(tensor_pruning_dependency.keys())
    eq_tensors_ids.sort()

    net_logger.info('Start Pruning.')
    net_pruner.prune(pruning_tensor_cfg)
    net_logger.info('Pruning Complete.')

    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model).cuda()
    else:
        net_model = torch.nn.DataParallel(net_model)

    net_model.training = True

    if args.optim == 'adamw':
        optimizer = optim.AdamW(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f'Unknown optimizer type {args.optim}')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    net_model.train()
    if isinstance(net_model, torch.nn.DataParallel):
        net_model.module.freeze_bn()
    else:
        net_model.freeze_bn()

    net_logger.info('Num Training Images: {}'.format(len(dataset_train)))

    if args.validation_only:
        net_model.eval()
        test(dataset_valid, net_model, start_epoch - 1, args, net_logger)
        print('Validation Done.')
        exit(0)

    for epoch_num in range(start_epoch, start_epoch + args.epochs):
        net_model.train()
        if isinstance(net_model, torch.nn.DataParallel):
            net_model.module.freeze_bn()
        else:
            net_model.freeze_bn()
        #net_model.module.freeze_bn()
        #net_model.eval()
        #test(dataset_valid, net_model, epoch_num, args, net_logger)
        #exit(0)

        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                # print('Image Shape : {}'.format(str(data['img'][0,:,:,:].shape)))
                # print(data['annot'])
                # exit(0)
                imgs  = data['img']
                annot = data['annot']
                if args.num_gpus == 1:
                    # if only one gpu, just send it to cuda:0
                    # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                    imgs = imgs.cuda()
                    annot = annot.cuda()
                classification_loss, regression_loss = net_model(imgs, annot, return_loss=True)

                #if torch.cuda.is_available():
                #    #classification_loss, regression_loss = net_model([data['img'].cuda().float(), data['annot']])
                #    classification_loss, regression_loss = net_model(data['img'].cuda().float(), data['annot'].cuda())
                #else:
                #    #classification_loss, regression_loss = net_model([data['img'].float(), data['annot']])
                #    classification_loss, regression_loss = net_model(data['img'].float(), data['annot'])
                    
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

                if epoch_num == 1 and (iter_num % 10 == 0):
                    _log = 'Epoch: {:>3} | Iter: {:>4} | Class loss: {:1.5f} | BBox loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))
                    net_logger.info(_log)
                elif(iter_num % 100 == 0):
                    _log = 'Epoch: {:>3} | Iter: {:>4} | Class loss: {:1.5f} | BBox loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))
                    net_logger.info(_log)

                del classification_loss
                del regression_loss
            except Exception as e:
                raise Exception
                #print(e)
                #continue


        # if (epoch_num + 1) % 1 == 0:
        #if (epoch_num + 1) % args.valid_period == 0:
        if epoch_num % args.valid_period == 0:
            test(dataset_valid, net_model, epoch_num, args, net_logger)


        scheduler.step(np.mean(epoch_loss))
        print('Learning Rate:', str(scheduler._last_lr))
        # save_checkpoint(net_model, os.path.join(
        #                 'saved', '{}_{}_{}.pt'.format(args.dataset, network_name, epoch_num)))

    net_logger.info('Training Complete.')

    net_model.eval()
    test(dataset_valid, net_model, epoch_num, args, net_logger)

    # save_checkpoint(net_model, os.path.join(
    #            'saved', '{}_{}_final_{}.pt'.format(args.dataset, network_name, epoch_num)))


def save_checkpoint(model, save_path):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, CustomDataParallel):
        print(f'model is DataParallel: {str(type(model))}')
        torch.save(model.module.state_dict(), save_path)
    else:
        print('model is Not DataParallel')
        torch.save(model.state_dict(), save_path)


def write_support_backbones(support_backbones):
    import json
    print('KSevenDet default models')
    support_backbone_arch = list(support_backbones.keys())
    support_backbone_arch.sort()
    for k in support_backbone_arch:
        print(f'  Backbone {k}')
        support_variant = list(support_backbones[k])
        support_variant.sort()
        support_backbones[k] = support_variant
        for m in support_variant:
            print(f'    {m}')
    save_path = os.path.join('support_backbones.json')
    with open(save_path, 'w') as outfile:
        json.dump(support_backbones, outfile, indent=4)
    print('done')



TENSOR_PRUNING_DEPENDENCY_JSON_FILE = 'ksevendet-resnet18-bifpn_tensor_pruning_dependency.json'

if __name__ == '__main__':
    tensor_pruning_dependency = json.load(open(TENSOR_PRUNING_DEPENDENCY_JSON_FILE))
    experiment_pruning_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    eq_tensors_ids = list(tensor_pruning_dependency.keys())
    eq_tensors_ids.sort()

    for pruning_rate in experiment_pruning_rate:
        pruning_tensor_cfg = list()
        for eq_t_id in eq_tensors_ids:
            pruning_args = {
                'pruning_type': 'random',
                'pruning_rate': pruning_rate,
            }
            pruning_tensor_cfg.append([eq_t_id, pruning_args])
        train_pruning_model(pruning_tensor_cfg, 
                            cfg_name='x{}_uniform'.format(int(pruning_rate * 100)),
                            tensor_pruning_dependency=tensor_pruning_dependency)
    
    print('done')
