import argparse
import collections
import numpy as np
from torchvision import transforms
from datasettool.dataloader import KSevenDataset, CocoDataset, FLIRDataset, collater, \
                                   Resizer, AspectRatioBasedSampler, Augmenter, \
                                   Normalizer
from torch.utils.data import DataLoader

import os
import sys
import logging
# from thop import profile
# from distiller import model_summaries

from PIL import Image, ImageDraw

import pdb

_COLORS = [
    '#0000ff',
    '#ffff00',
]

def get_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    
    parser.add_argument('--dataset', help='Dataset Name')
    parser.add_argument('--dataset_root', default='/root/data/',
                        help='Dataset root directory path [/root/data/coco/, /root/data/FLIR_ADAS]')
    parser.add_argument('--dataset_type', default='kseven',
                        help='Dataset type, must be one of kseven, coco, flir')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size for training')
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images (height, width)')
    parser.add_argument('--resize_mode', default=1, type=int,
                        help='The resize mode for Resizer')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--output_path', default='unit_test/augmentation', type=str,
                        help='Output path')
    parser.add_argument('--output_name', default=None, type=str,
                        help='Output name')

    args = parser.parse_args()
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

    return my_logger


def main():
    args = get_args()
    assert args.dataset, 'dataset must provide'

    net_logger = get_logger(name='Network Logger', args=args)
    net_logger.info('Dataset Name: {}'.format(args.dataset))
    net_logger.info('Dataset Root: {}'.format(args.dataset_root))
    net_logger.info('Dataset Type: {}'.format(args.dataset_type))

    _shape_1, _shape_2 = tuple(map(int, args.input_shape.split(',')))
    _augmenter  = Augmenter(use_flip=True, use_noise=True, use_brightness=True, use_scale=True)
    # _augmenter  = Augmenter(use_flip=False, use_noise=True, noise_theta=1.0, use_brightness=False, use_scale=False)
    # _augmenter  = Augmenter(use_flip=False, use_noise=False, use_brightness=True, brightness_theta=1.0, use_scale=False)
    # _augmenter  = Augmenter(use_flip=False, use_noise=False, use_brightness=False, use_scale=True, scale_theta=1.0)
    _normalizer = Normalizer()
    if args.resize_mode == 0:
        _resizer = Resizer(min_side=_shape_1, max_side=_shape_2, resize_mode=args.resize_mode, logger=net_logger)
    elif args.resize_mode == 1:
        _resizer = Resizer(height=_shape_1, width=_shape_2, resize_mode=args.resize_mode, logger=net_logger)
    else:
        raise ValueError('Illegal resize mode.')
    transfrom_funcs_train = [
        _augmenter,
        _resizer,
    ]
    transfrom_funcs_valid = [
        _normalizer,
        _resizer,
    ]
    # Create the data loaders
    if args.dataset_type == 'kseven':
        # dataset_train = KSevenDataset(args.dataset_root, set_name='train', transform=transforms.Compose(transfrom_funcs_train))
        dataset_train = KSevenDataset(args.dataset_root, set_name='valid', transform=transforms.Compose(transfrom_funcs_train))
        # dataset_valid = KSevenDataset(args.dataset_root, set_name='train', transform=transforms.Compose(transfrom_funcs_valid))
        #dataset_valid = KSevenDataset(args.dataset_root, set_name='valid', transform=transforms.Compose(transfrom_funcs_valid))
    elif args.dataset_type == 'coco':
        dataset_train = CocoDataset(args.dataset_root, set_name='train', transform=transforms.Compose(transfrom_funcs_train))
        #dataset_valid = CocoDataset(args.dataset_root, set_name='valid', transform=transforms.Compose(transfrom_funcs_valid))
    else:
        raise ValueError('Dataset type not understood (must be FLIR, COCO or csv), exiting.')
    
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  shuffle=False,
                                  collate_fn=collater,
                                  pin_memory=True)
    
    net_logger.info('Number of Classes: {:>3}'.format(dataset_train.num_classes()))
    
    net_logger.info('Num Train Images: {}'.format(len(dataset_train)))

    counter_ = 0
    for iter_num, data in enumerate(dataloader_train):
        imgs  = data['img'].data.numpy()
        annots = data['annot'].data.numpy()

        imgs *= 255.0
        # imgs.astype(np.uint8)
        # print(imgs)

        # annotation format (x1, y1, x2, y2)
        print(f'Iter Num {iter_num}')
        # print(annots)
        # print(annots.shape)

        for i in range(len(imgs)):
            img = imgs[i,...]
            # print(f'max: {np.max(img)}')
            # print(f'min: {np.min(img)}')
            annot = annots[i,...]
            img_save_path = os.path.join(args.output_path, 'sample-{:06}.png'.format(counter_))
            #print(img.shape)
            #pdb.set_trace()

            # aa = img.data.numpy()
            # aa *= 255.0
            # aa = aa.transpose((1, 2, 0)).astype(np.uint8)
            aa = img.transpose((1, 2, 0)).astype(np.uint8)
            # print(aa.shape)
            im = Image.fromarray(aa).convert('RGB')
            im_draw = ImageDraw.Draw(im)
            for j in range(len(annot)):
                ann = annot[j]
                # ann = ann.data.numpy()
                # print(ann.shape)
                if ann[-1] == -1.0:
                    break
                x1, y1, x2, y2, _ = ann
                im_draw.rectangle(tuple([int(x1), int(y1), int(x2), int(y2)]), width = 2, outline = _COLORS[int(ann[-1])])

            im.save(img_save_path, 'PNG')
            # exit(0)
            counter_ += 1


if __name__ == '__main__':
    main()
