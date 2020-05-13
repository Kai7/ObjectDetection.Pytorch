import argparse
import yaml
import collections
import numpy as np
import torch
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
from datasettool.dataloader import KSevenDataset, CocoDataset, FLIRDataset, collater, \
                                   Resizer, AspectRatioBasedSampler, Augmenter, \
                                   Normalizer
from torch.utils.data import DataLoader
from datasettool import coco_eval
from datasettool.ksevendata_eval import coco_evaluate_ksevendata
#from datasettool.flir_eval import evaluate_flir

import skimage.draw
import skimage.io
import skimage.color
from PIL import Image, ImageDraw
import cv2

import matplotlib.pyplot as plt 

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

assert torch.__version__.split('.')[0] == '1'

import pdb

print('CUDA available: {}'.format(torch.cuda.is_available()))

CONVERT_FILE_LIMIT = 20*60*60
COLOR_LABEL = ['#00ff00',]
label_color_map = {
    'face_noMask'    : '#0000ff',
    'face_Mask'      : '#ffff00',
    'face'           : '#00ff00',
}


#parser = argparse.ArgumentParser(description='Simple testing script.')
#
#parser.add_argument('--dataset', help='Dataset type.')
#parser.add_argument('--architecture', default='RetinaNet', type=str,
#                    help='Network Architecture.')
#parser.add_argument('--resume', type=str,
#                    help='Checkpoint state_dict file to resume training from')
#parser.add_argument('--num_classes', type=int,
#                    help='The number of class.')
#parser.add_argument('--demo_path', type=str,
#                    help='Demo images directory path.')
#parser.add_argument('--threshold', default=0.6, type=float,
#                    help='Object pasitive threshold.')
#parser.add_argument('--output_path', default='video', type=str,
#                    help='Output path')

def get_args():
    parser = argparse.ArgumentParser(description='Inference input images and transfer result to video.')
    
    parser.add_argument('--dataset', help='Dataset Name')
    # parser.add_argument('--dataset_root', default='/root/data/',
    #                     help='Dataset root directory path [/root/data/coco/, /root/data/FLIR_ADAS]')
    # parser.add_argument('--dataset_type', default='kseven',
    #                     help='Dataset type, must be one of kseven, coco, flir')
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
    parser.add_argument('--input_shape', default='512,512', type=str,
                        help='Input images (height, width)')
    parser.add_argument('--num_classes', type=int,
                        help='The number of class.')
    parser.add_argument('--demo_path', type=str,
                        help='Demo images directory path.')
    parser.add_argument('--threshold', default=0.6, type=float,
                        help='Object pasitive threshold.')
    parser.add_argument('--output_path', default='video', type=str,
                        help='Output path')
    parser.add_argument('--output_name', default=None, type=str,
                        help='Output name')
    parser.add_argument('--resize_mode', default=1, type=int,
                        help='The resize mode for Resizer')

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

    return my_logger

def main():
    args = get_args()
    assert args.dataset, 'dataset must provide'
    if args.resume is None:
        raise ValueError('Must provide --resume when testing.')

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
            assert 0, 'not support now.'
            assert isinstance(ksevendet_cfg, dict)
            network_name = f'{args.architecture}-{ksevendet_cfg["backbone"]}_specifical-{ksevendet_cfg["neck"]}'
    elif args.architecture in support_architectures:
        network_name = args.architecture
    else:
        raise ValueError('Architecture {} is not support.'.format(args.architecture))

    args.network_name = network_name
    net_logger = get_logger(name='Network Logger', args=args)
    net_logger.info('Positive Threshold: {:.2f}'.format(args.threshold))

    _shape_1, _shape_2 = tuple(map(int, args.input_shape.split(',')))
    _normalizer = Normalizer(inference_mode=True)
    if args.resize_mode == 0:
        _resizer = Resizer(min_side=_shape_1, max_side=_shape_2, resize_mode=args.resize_mode, logger=net_logger, inference_mode=True)
    elif args.resize_mode == 1:
        _resizer = Resizer(height=_shape_1, width=_shape_2, resize_mode=args.resize_mode, logger=net_logger, inference_mode=True)
    else:
        raise ValueError('Illegal resize mode.')

    transfrom_funcs_valid = [
        _normalizer,
        _resizer,
    ]
    transform = transforms.Compose(transfrom_funcs_valid)
    
    net_logger.info('Number of Classes: {:>3}'.format(args.num_classes))

    build_param = {'logger': net_logger}
    if args.architecture == 'ksevendet':
        net_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
    elif args.architecture == 'retinanet-p45p6':
        net_model = retinanet.retinanet_p45p6(num_classes=args.num_classes, **build_param)
    elif args.architecture.split('-')[0] == 'retinanet':
        net_model = retinanet.build_retinanet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
    elif args.architecture.split('-')[0] == 'efficientdet':
        net_model = efficientdet.build_efficientdet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
    else:
        assert 0, 'architecture error'

    net_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
    net_model.load_state_dict(torch.load(args.resume))
    #model = torch.load(args.resume)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            net_model = net_model.cuda()

    if torch.cuda.is_available():
       net_model = torch.nn.DataParallel(net_model).cuda()
    else:
       net_model = torch.nn.DataParallel(net_model)

    demo_image_files = os.listdir(args.demo_path)
    demo_image_files.sort()
    #if len(demo_image_files) > CONVERT_FILE_LIMIT:
    #    print('WARNING: Too many files...    total {} files.'.format(len(demo_image_files)))

    net_model.eval()

    img_array = []
    # print(net_model)


    for f in demo_image_files:
    #for f in demo_image_files[:1]:
    # for f in demo_image_files[:100]:
    #for f in demo_image_files[:min(len(demo_image_files), CONVERT_FILE_LIMIT)]:
        print(f)
        if f[-3:] not in ['png', 'jpg']:
            continue
        #img = skimage.io.imread(os.path.join(args.demo_path, f))
        #if len(img.shape) == 2:
        #    img = skimage.color.gray2rgb(img)
        #print(np.sum(img - a_pil_img))
        img = Image.open(os.path.join(args.demo_path, f)).convert('RGB')
        a_img = np.array(img)
        # print(a_img)
        a_img = a_img.astype(np.float32) / 255.0
        # print(a_img.shape)
        a_img = transform(a_img)
        # print(a_img.shape)
        a_img = torch.unsqueeze(a_img, 0)
        # print(a_img.shape)
        a_img = a_img.permute(0, 3, 1, 2)
        # print(a_img.shape)

        # print('predict...')
        scores, labels, boxes = net_model(a_img, return_loss=False)

        scores = scores.cpu()
        labels = labels.cpu()
        boxes  = boxes.cpu()

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

        #if args.dataset == 'thermal':
        #    img = img.resize((80, 60))

        draw = ImageDraw.Draw(img)
        for box_id in range(boxes.shape[0]):
            score = float(scores[box_id])
            label = int(labels[box_id])
            box = boxes[box_id, :]

            # scores are sorted, so we can break
            if score < args.threshold:
                break

            x, y, w, h = box
            color_ = COLOR_LABEL[label]
            _text_offset_x, _text_offset_y = 2, 3
            #draw.rectangle(tuple([x, y, x+w, y+h]), width = 1, outline ='green')
            draw.rectangle(tuple([x, y, x+w, y+h]), width = 1, outline = color_)
            draw.text(tuple([int(x)+_text_offset_x, int(y)+_text_offset_y]),
                      '{:.4f}'.format(score), fill = color_)
            
            # append detection to results
            # results.append(image_result)
        #plt.figure()
        #plt.imshow(img)
        #plt.axis('off')
        #plt.show()
        img_array.append(np.array(img))

    
    height, width, layers = img_array[0].shape
    size = (width, height)
    fps = 25
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    out_video_file = os.path.join(args.output_path, 
                                  '{}.avi'.format(os.path.basename(args.demo_path) if not args.output_name else args.output_name))
    print('Convert to video... {}'.format(out_video_file))
    out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
        
    print('Done')
    # print(demo_image_files)
    #net_logger.info('There are {} images for testing.'.format(len(dataset_valid)))
    #net_logger.info('Testing start ...')
    #net_model.is_training = False
    #with torch.no_grad():
    #    if(parser.dataset == 'VOC'):
    #        evaluate(dataset, net_model)
    #    elif parser.dataset == 'COCO':
    #        evaluate_coco(dataset, net_model)
    #    elif parser.dataset == 'FLIR':
    #        summarize = evaluate_flir(dataset_valid, net_model)
    #        net_logger.info('\n{}'.format(summarize))
    #    elif parser.dataset == 'thermal':
    #        summarize = coco_evaluate_cvidata(dataset_valid, net_model)
    #        net_logger.info('\n{}'.format(summarize))
    #        
    #    else:
    #        print('ERROR: Unknow dataset.')


if __name__ == '__main__':
    main()
