from functools import partial

import argparse
import yaml
import collections
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from architectures import retinanet, efficientdet
# from architectures import ksevendet
from ksevendet.architecture import ksevendet
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
# from thop import profile
# from distiller import model_summaries
from distiller import model_find_module_name

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
        filehandler = logging.FileHandler(os.path.join('log', '{}_{}.log'.format(args.dataset, args.network_name)), mode='a')
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


def main():
    args = get_args()
    assert args.dataset, 'dataset must provide'
    # pdb.set_trace()
    default_support_backbones = registry._module_to_models
    
    print(f'args.epochs = {args.epochs}')
    print(f'args.batch_size = {args.batch_size}')

    # write_support_backbones(default_support_backbones)

    # BACKBONE = 'shufflenetv2'
    # VARIANT  = 'shufflenetv2_x0_5'
    # BACKBONE = 'densenet'
    # BACKBONE = 'mnasnet'
    # BACKBONE = 'mobilenetv2'
    # BACKBONE = 'resnet'
    # BACKBONE = 'res2net'
    # BACKBONE = 'sknet'

    # NECK = 'fpn'
    # NECK = 'panet-fpn'
    NECK = 'bifpn'

    # FPN_FEATURES_NUM = 256
    FPN_FEATURES_NUM = 64

    support_architectures = [
        'ksevendet',
    ]
    support_architectures += [f'efficientdet-d{i}' for i in range(8)]
    support_architectures += [f'retinanet-res{i}' for i in [18, 34, 50, 101, 152]]

    support_architectures.append('retinanet-p45p6')

    print(support_architectures)

    if args.architecture == 'ksevendet':
        # ksevendet_cfg = {
        #     'backbone'          : BACKBONE,
        #     'variant'           : VARIANT,
        #     'neck'              : NECK,
        #     'fpn_features_num'  : FPN_FEATURES_NUM,
        #     'backbone_feature_pyramid_levels' : [3, 4, 5],
        #     'neck_feature_pyramid_levels'     : [3, 4, 5, 6, 7],
        # }
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

    net_logger = get_logger(name='Network Logger', args=args)
    net_logger.info('Network Name: {}'.format(network_name))
    net_logger.info('Dataset Name: {}'.format(args.dataset))
    net_logger.info('Dataset Root: {}'.format(args.dataset_root))
    net_logger.info('Dataset Type: {}'.format(args.dataset_type))

    height, width = _shape_1, _shape_2 = tuple(map(int, args.input_shape.split(',')))
    _normalizer = Normalizer()
    #_augmenter  = Augmenter(scale_min=0.9, logger=net_logger)
    _augmenter  = Augmenter(use_scale=False, scale_min=0.9, logger=net_logger)
    # exit(0)
    if args.resize_mode == 0:
        _resizer = Resizer(min_side=_shape_1, max_side=_shape_2, resize_mode=args.resize_mode, logger=net_logger)
    elif args.resize_mode == 1:
        _resizer = Resizer(height=_shape_1, width=_shape_2, resize_mode=args.resize_mode, logger=net_logger)
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
    # pdb.set_trace()
    
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

    # _shape_1, _shape_2 = tuple(map(int, args.input_shape.split(',')))
    # sample_input = torch.randn(1, 3, _shape_1, _shape_2)
    # sample_input_shape = dataset_valid[0]['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape
    # net_model.eval()
    # model_summaries.model_summary(net_model, 'compute', input_shape=sample_input_shape)
    # exit(0)

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
        assert 0, "resume file most to be specify."

    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()
        

    sample_image = np.zeros((height, width, 3)).astype(np.float32)
    sample_image = torch.from_numpy(sample_image)
    sample_input = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
    sample_input_shape = sample_image.permute(2, 0, 1).cuda().float().unsqueeze(dim=0).shape

    def conv_hook(module, fea_in, fea_out, model):
        # module_name.append(module.__class__.__name__)
        # features_in_hook.append(fea_in[0].shape)
        # features_out_hook.append(fea_out[0].shape)

        # conv_weights.append(module.weight)
        mod_name = model_find_module_name(model, module)
        print('find Conv module: {}'.format(mod_name))
        conv_modules_map[mod_name] = module

        return None

    def bn_hook(module, fea_in, fea_out, model):
        # module_name.append(module.__class__.__name__)
        # features_in_hook.append(fea_in[0].shape)
        # features_out_hook.append(fea_out[0].shape)

        mod_name = model_find_module_name(model, module)
        print('find BN module: {}'.format(mod_name))
        bn_modules_map[mod_name] = module

        return None

    def module_io_shape_recorder(m):
        if isinstance(m, torch.nn.Conv2d):
            hook_handles.append(m.register_forward_hook(partial(conv_hook, model=net_model)))
        if isinstance(m, torch.nn.BatchNorm2d):
            hook_handles.append(m.register_forward_hook(partial(bn_hook, model=net_model)))

    conv_modules_map = collections.OrderedDict()
    bn_modules_map = collections.OrderedDict()
    hook_handles = []
    with torch.no_grad():
        net_model.apply(module_io_shape_recorder)
        net_model(sample_input, return_head=True)
        # Unregister from the forward hooks
        print(len('remove hook'))
        for handle in hook_handles:
            handle.remove()
    
        net_model(sample_input, return_head=True)
        print(len('remove hook model inference done.'))
    
    print(type(conv_modules_map.keys()))
    conv_modules = list(conv_modules_map.keys())
    for i in range(10):
        print(f'[{i}] {conv_modules[i]}')
    print(type(bn_modules_map.keys()))
    bn_modules = list(bn_modules_map.keys())
    for i in range(10):
        print(f'[{i}] {bn_modules[i]}')
    

    pruning_channel_conv_layers = [
        conv_modules[1],   # backbone.conv1
    ]
    pruning_channel_to_filter_conv_chain = {
        conv_modules[1] : [
            conv_modules[0],    # backbone.layer1.0.conv1
            conv_modules[2],    # backbone.layer1.0.conv2
            conv_modules[4],    
        ] ,
        conv_modules[3] : [
            conv_modules[0],    # backbone.layer1.0.conv1
            conv_modules[2],    # backbone.layer1.0.conv2
            conv_modules[4],    
        ] ,
        conv_modules[5] : [
            conv_modules[0],    # backbone.layer1.0.conv1
            conv_modules[2],    # backbone.layer1.0.conv2
            conv_modules[4],    
        ] ,
        conv_modules[7] : [
            conv_modules[0],    # backbone.layer1.0.conv1
            conv_modules[2],    # backbone.layer1.0.conv2
            conv_modules[4],    
        ] ,
    }
    pruning_filter_to_channel_conv_chain = {
        conv_modules[0] : [
            conv_modules[1],    # backbone.layer1.0.conv1
            conv_modules[3],    # backbone.layer1.0.conv2
            conv_modules[5],    
            conv_modules[7],    
        ] ,
        conv_modules[2] : [
            conv_modules[1],    # backbone.layer1.0.conv1
            conv_modules[3],    # backbone.layer1.0.conv2
            conv_modules[5],    
            conv_modules[7],    
        ] ,
        conv_modules[4] : [
            conv_modules[1],    # backbone.layer1.0.conv1
            conv_modules[3],    # backbone.layer1.0.conv2
            conv_modules[5],    
            conv_modules[7],    
        ] ,
    }
    conv_bn_chain = {
        conv_modules[0] : bn_modules[0],  # backbone.conv1 : backbone.bn1
        conv_modules[1] : bn_modules[1],  # backbone.layer1.0.conv1 : backbone.layer1.0.bn1
        conv_modules[2] : bn_modules[2],  # backbone.layer1.0.conv2 : backbone.layer1.0.bn2
        conv_modules[3] : bn_modules[3],  
        conv_modules[4] : bn_modules[4],  
        conv_modules[5] : bn_modules[5],  
        conv_modules[7] : bn_modules[7],  
    }

    print('Pruning Channel to Filter Conv Chain')
    for k in pruning_channel_to_filter_conv_chain:
        print(f'> {k}')
        for c in  pruning_channel_to_filter_conv_chain[k]:
            print(f'  - {c}')

    print('Pruning Filter to Channel Conv Chain')
    for k in pruning_filter_to_channel_conv_chain:
        print(f'> {k}')
        for c in  pruning_filter_to_channel_conv_chain[k]:
            print(f'  - {c}')

    def is_conv_chain_valid(conv_module_1, conv_module_2):
        w1_shape = conv_module_1.weight.shape
        w2_shape = conv_module_2.weight.shape
        return w1_shape[0] == w2_shape[1]

    def prune_conv_channel(module_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=7):
        assert not conv_modules_map[module_name].bias, 'not support'
        print('> Pruning Channels : {}'.format(module_name))
        conv_module = conv_modules_map[module_name]
        conv_weight = conv_module.weight
        w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_weight.shape
        print('    original weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

        rand_tensor = torch.rand((filter_num, channel_num - pruning_num, kernel_h, kernel_w))
        rand_param  = torch.nn.Parameter(rand_tensor)
        conv_module.weight = rand_param
        torch.nn.init.kaiming_normal_(conv_module.weight, mode='fan_out', nonlinearity='relu')
        w_shape = filter_num, channel_num, kernel_h, kernel_w = rand_param.shape
        print('     pruning weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))
        #conv_bias = conv_module.bias

        for m_name in pruning_channel_to_filter_conv_chain[module_name]:
            if not is_conv_chain_valid(conv_modules_map[m_name], conv_module):
                prune_conv_filter(m_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=pruning_num)



    def prune_conv_filter(module_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=7):
        assert not conv_modules_map[module_name].bias, 'not support'
        print('> Pruning Filters : {}'.format(module_name))
        conv_module = conv_modules_map[module_name]
        conv_weight = conv_module.weight
        w_shape = filter_num, channel_num, kernel_h, kernel_w = conv_weight.shape
        print('    original weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

        rand_tensor = torch.rand((filter_num - pruning_num, channel_num, kernel_h, kernel_w))
        rand_param  = torch.nn.Parameter(rand_tensor)
        conv_module.weight = rand_param
        torch.nn.init.kaiming_normal_(conv_module.weight, mode='fan_out', nonlinearity='relu')
        w_shape = filter_num, channel_num, kernel_h, kernel_w = rand_param.shape
        print('     pruning weight shape = [{:>4},{:>4},{:>4},{:>4}]'.format(*w_shape))

        bn_module_name = conv_bn_chain.get(module_name, None)
        if not bn_module_name:
            return
        print(' - Pruning BN Channels : {}'.format(bn_module_name))
        print('     init BN, channels num = {}'.format(filter_num))
        bn_module = bn_modules_map[bn_module_name]
        #print(_num_filters, bn_module.eps, bn_module.momentum, bn_module.affine, bn_module.track_running_stats)
        bn_module.__init__(filter_num, bn_module.eps, bn_module.momentum, bn_module.affine, bn_module.track_running_stats)

        for m_name in pruning_filter_to_channel_conv_chain[module_name]:
            if not is_conv_chain_valid(conv_module, conv_modules_map[m_name]):
                prune_conv_channel(m_name, conv_modules_map, bn_modules_map, conv_bn_chain, pruning_num=pruning_num)
        


    for conv_layer in pruning_channel_conv_layers:
        prune_conv_channel(conv_layer, conv_modules_map, bn_modules_map, conv_bn_chain) 

    print('Pruning Done.')

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

    #dummy_input = torch.randn(1, 3, _shape_1, _shape_2, device='cuda')
    #out = net_model(dummy_input, return_head=True)
    #print('Done')
    #exit(0)

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

                if epoch_num == 0 and (iter_num % 10 == 0):
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
        save_checkpoint(net_model, os.path.join(
                        'saved', '{}_{}_{}.pt'.format(args.dataset, network_name, epoch_num)))

    net_logger.info('Training Complete.')

    net_model.eval()
    test(dataset_valid, net_model, epoch_num, args, net_logger)

    save_checkpoint(net_model, os.path.join(
               'saved', '{}_{}_final_{}.pt'.format(args.dataset, network_name, epoch_num)))
    # torch.save(net_model.module.state_dict(), 'model_final.pt')


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

if __name__ == '__main__':
    main()
