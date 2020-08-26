import argparse
import json
import yaml
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
import ksevendet.architecture.backbone.registry as registry
from ksevendet.ksevenpruner import KSevenPruner
from datasettool.dataloader import KSevenDataset, CocoDataset, FLIRDataset, collater, \
                                   Resizer, AspectRatioBasedSampler, Augmenter, \
                                   Normalizer, ToTorchTensor
from torch.utils.data import DataLoader
from datasettool import coco_eval
from datasettool.flir_eval import evaluate_flir
from datasettool.coco_eval import evaluate_coco
from datasettool.ksevendata_eval import coco_evaluate_ksevendata

from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights

import os
import sys
import logging
# from thop import profile
# from distiller import model_summaries

import pdb

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

KSEVEN_MODEL = 'ksevendet'

def get_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a object detection network.')
    
    parser.add_argument('--dataset', help='Dataset Name')
    parser.add_argument('--dataset_root', default='/root/data/',
                        help='Dataset root directory path [/root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--dataset_type', default='kseven',
                        help='Dataset type, must be one of kseven, coco, flir')
    parser.add_argument('--model_config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')
    parser.add_argument('--tensor_dependency', default=None, type=str,
                        help='Tensor pruning dependency json file path.')
    parser.add_argument('--pruning_rate', default=0.0, type=float,
                        help='Uniform pruning rate.')
    parser.add_argument('--pruning_config', default=None, type=str,
                        help='Special pruning config path.')
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

    assert args.dataset, 'dataset must provide'

    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_cfg = yaml.safe_load(f)
        setattr(args, 'architecture', model_cfg.pop('architecture'))
        setattr(args, 'model_cfg', model_cfg)

    support_architectures = [
        KSEVEN_MODEL,
    ]
    # support_architectures += [f'efficientdet-d{i}' for i in range(8)]
    # support_architectures += [f'retinanet-res{i}' for i in [18, 34, 50, 101, 152]]
    print("Support Architectures:")
    print(support_architectures)

    if args.architecture == KSEVEN_MODEL:
        model_cfg = args.model_cfg
        if model_cfg.get('variant'):
            network_name = f'{args.architecture}-{args.model_cfg["variant"]}-{args.model_cfg["neck"]}'
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


def train():
    args = get_args()

    # default_support_backbones = registry._module_to_models

    train_logger = get_logger(name='ksevendet_train_logger', args=args)
    train_logger.info('Network Name: {}'.format(args.network_name))
    train_logger.info('Dataset Name: {}'.format(args.dataset))
    train_logger.info('Dataset Root: {}'.format(args.dataset_root))
    train_logger.info('Dataset Type: {}'.format(args.dataset_type))
    train_logger.info('Epochs      : {}'.format(args.epochs))
    train_logger.info('Batch Size  : {}'.format(args.batch_size))
    train_logger.info('Weight Decay  : {}'.format(args.weight_decay))
    train_logger.info('Learning Rate : {}'.format(args.lr))

    #_augmenter  = Augmenter(scale_min=0.9, logger=train_logger)
    _augmenter  = Augmenter(use_scale=False, scale_min=0.9, logger=train_logger)
    _resizer = Resizer(height=args.height, width=args.height, resize_mode=args.resize_mode, logger=train_logger)
    _normalizer = Normalizer()
    _totensor = ToTorchTensor()
    transfrom_funcs_train = [
        _augmenter,
        _resizer,
        _normalizer,
        _totensor,
    ]
    transfrom_funcs_valid = [
        _resizer,
        _normalizer,
        _totensor,
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

    train_logger.info('Num Classes : {:>2}'.format(dataset_train.num_classes()))
    train_logger.info('Num Train Images : {:>7}'.format(len(dataset_train)))
    train_logger.info('Num Valid Images : {:>7}'.format(len(dataset_valid)))

    build_param = {'logger': train_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(args.model_cfg, num_classes=dataset_train.num_classes(), pretrained=False, **build_param)
    elif args.architecture.split('-')[0] == 'retinanet':
        net_model = retinanet.build_retinanet(args.architecture, num_classes=dataset_train.num_classes(), pretrained=False, **build_param)
    elif args.architecture.split('-')[0] == 'efficientdet':
        net_model = efficientdet.build_efficientdet(args.architecture, num_classes=dataset_train.num_classes(), pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    use_gpu = True
    if use_gpu and torch.cuda.is_available():
        net_model = net_model.cuda()

    # sample_image = np.zeros((args.height, args.width, 3)).astype(np.float32)
    # sample_image = torch.from_numpy(sample_image)
    # sample_input = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    # sample_input_shape = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape
    # # the following statement is unnecessary.
    # net_model.set_onnx_convert_info(fixed_size=(height, width))

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

        train_logger.info('Start Pruning.')
        net_pruner.prune(args.pruning_tensor_cfg)
        train_logger.info('Pruning Complete.')

    # load last weights
    if args.resume is not None:
        train_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
        try:
            ret = net_model.load_state_dict(torch.load(args.resume), strict=False)
        except RuntimeError as e:
            train_logger.warning(f'Ignoring {e}')
            train_logger.warning('Don\'t panic if you see this,')
            train_logger.warning('  this might be because you load a pretrained weights with different number of classes.')
            train_logger.warning('The rest of the weights should be loaded already.')

        s_b = args.resume.rindex('_')
        s_e = args.resume.rindex('.')
        start_epoch = int(args.resume[s_b+1:s_e]) + 1
        train_logger.info('Continue on {} Epoch'.format(start_epoch))
    else:
        start_epoch = 1
        
    ## for EfficientDet 
    ## freeze backbone if train head_only
    #if opt.head_only:
    #    def freeze_backbone(m):
    #        classname = m.__class__.__name__
    #        for ntl in ['EfficientNet', 'BiFPN']:
    #            if ntl in classname:
    #                for param in m.parameters():
    #                    param.requires_grad = False
    #    model.apply(freeze_backbone)
    #    print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    #if args.num_gpus > 1 and args.batch_size // args.num_gpus < 4:
    #    net_model.apply(replace_w_sync_bn)

    #if args.num_gpus > 0:
    #    net_model = net_model.cuda()
    #    if args.num_gpus > 1:
    #        net_model = CustomDataParallel(net_model, args.num_gpus)

    #if args.num_gpus > 0 and torch.cuda.is_available():
    #    net_model = net_model.cuda()

    #if args.num_gpus > 1:
    #    net_model = torch.nn.DataParallel(net_model).cuda()


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
        optimizer = optim.SGD(net_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f'Unknown optimizer type {args.optim}')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    net_model.train()

    if isinstance(net_model, torch.nn.DataParallel):
        net_model.module.freeze_bn()
    else:
        net_model.freeze_bn()

    if args.validation_only:
        net_model.eval()
        test(dataset_valid, net_model, start_epoch - 1, args, train_logger)
        print('Validation Done.')
        exit(0)

    for epoch_num in range(start_epoch, start_epoch + args.epochs):
        if start_epoch != 1:
            test(dataset_valid, net_model, epoch_num, args, train_logger)

        net_model.train()
        if isinstance(net_model, torch.nn.DataParallel):
            net_model.module.freeze_bn()
        else:
            net_model.freeze_bn()

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

                if epoch_num == 0 and (iter_num % 10 == 0):
                    _log = 'Epoch: {:>3} | Iter: {:>4} | Class loss: {:1.5f} | BBox loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))
                    train_logger.info(_log)
                elif(iter_num % 100 == 0):
                    _log = 'Epoch: {:>3} | Iter: {:>4} | Class loss: {:1.5f} | BBox loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))
                    train_logger.info(_log)

                del classification_loss
                del regression_loss
            except Exception as e:
                raise Exception

        if epoch_num % args.valid_period == 0:
            test(dataset_valid, net_model, epoch_num, args, train_logger)

        scheduler.step(np.mean(epoch_loss))
        # print('Learning Rate:', str(scheduler._last_lr))
        save_checkpoint(net_model, os.path.join('saved', '{}_{}{}_{}.pt'.format(
                                                args.dataset, args.network_name,
                                                '_' + args.cfg_name if args.cfg_name else '', epoch_num)))

    train_logger.info('Training Complete.')

    # net_model.eval()
    test(dataset_valid, net_model, epoch_num, args, train_logger)

    # save_checkpoint(net_model, os.path.join(
    #            'saved', '{}_{}_final_{}.pt'.format(args.dataset, network_name, epoch_num)))
    save_checkpoint(net_model, os.path.join('saved', '{}_{}{}_final_{}.pt'.format(
                                            args.dataset, args.network_name,
                                            '_' + args.cfg_name if args.cfg_name else '', epoch_num)))


def save_checkpoint(model, save_path):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, CustomDataParallel):
        print(f'model is DataParallel: {str(type(model))}')
        torch.save(model.module.state_dict(), save_path)
    else:
        print('model is Not DataParallel')
        torch.save(model.state_dict(), save_path)


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
        # filehandler = logging.FileHandler(os.path.join('log', '{}_{}.log'.format(args.dataset, args.network_name)), mode='a')
        filehandler = logging.FileHandler(os.path.join('log', '{}_{}{}.log'.format(
                                                       args.dataset, args.network_name,
                                                       '_' + args.cfg_name if args.cfg_name else '')), mode='a')
        filehandler.setFormatter(formatter)
        my_logger.addHandler(filehandler)

    return my_logger


def test(dataset, model, epoch, args, logger=None):
    logger.info("{} epoch: \t start validation....".format(epoch))
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

if __name__ == '__main__':
    train()

    print('\ndone')