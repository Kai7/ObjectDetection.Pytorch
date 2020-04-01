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

def main():
    parser = argparse.ArgumentParser(description='Simple testing script.')

    parser.add_argument('--dataset', help='Dataset type.')
    parser.add_argument('--resume', type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--demo_path', type=str,
                        help='Demo images directory path.')
    parser.add_argument('--threshold', default=0.6, type=float,
                        help='Object pasitive threshold.')
    parser.add_argument('--output_path', default='video', type=str,
                        help='Output path')

    args = parser.parse_args()

    if args.dataset == 'thermal':
        transform=transforms.Compose([Normalizer(demo=True), 
                                      Resizer(min_side=int(60), max_side=int(80), demo=True)])
    else:
        raise ValueError('unknow demo size.')

    # print('network_name:', network_name)
    net_logger    = logging.getLogger('Demo Logger')
    formatter     = logging.Formatter(LOGGING_FORMAT)
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    net_logger.addHandler(streamhandler)
    net_logger.setLevel(logging.INFO)

    net_logger.info('Positive Threshold: {:.2f}'.format(args.threshold))

    
    build_param = {'logger': net_logger}
    if args.resume is not None:
        net_logger.info('Loading Checkpoint : {}'.format(args.resume))
        model = torch.load(args.resume)
    else:
        raise ValueError('Must provide --resume when testing.')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            model = model.cuda()

    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    #     model = torch.nn.DataParallel(model)

    demo_image_files = os.listdir(args.demo_path)
    demo_image_files.sort()
    if len(demo_image_files) > CONVERT_FILE_LIMIT:
        print('WARNING: Too many files...    total {} files.'.format(len(demo_image_files)))

    model.eval()

    img_array = []
    # print(model)


    #for f in demo_image_files:
    #for f in demo_image_files[:1]:
    # for f in demo_image_files[:100]:
    for f in demo_image_files[:min(len(demo_image_files), CONVERT_FILE_LIMIT)]:
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
        scores, labels, boxes = model(a_img)

        scores = scores.cpu()
        labels = labels.cpu()
        boxes  = boxes.cpu()

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]

        img = img.resize((80, 60))
        draw = ImageDraw.Draw(img)
        for box_id in range(boxes.shape[0]):
            score = float(scores[box_id])
            label = int(labels[box_id])
            box = boxes[box_id, :]

            # scores are sorted, so we can break
            if score < args.threshold:
                break

            x, y, w, h = box
            draw.rectangle(tuple([x, y, x+w, y+h]), width = 1, outline ='green')
            
            # append detection to results
            # results.append(image_result)
        #plt.figure()
        #plt.imshow(img)
        #plt.axis('off')
        #plt.show()
        img_array.append(np.array(img))

    
    print('Convert to video...')
    height, width, layers = img_array[0].shape
    size = (width, height)
    fps = 20
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out_video_file = os.path.join(args.output_path, '{}.avi'.format(os.path.basename(args.demo_path)))
    out = cv2.VideoWriter(out_video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
        
    print('Done')
    # print(demo_image_files)
    #net_logger.info('There are {} images for testing.'.format(len(dataset_valid)))
    #net_logger.info('Testing start ...')
    #model.is_training = False
    #with torch.no_grad():
    #    if(parser.dataset == 'VOC'):
    #        evaluate(dataset, model)
    #    elif parser.dataset == 'COCO':
    #        evaluate_coco(dataset, model)
    #    elif parser.dataset == 'FLIR':
    #        summarize = evaluate_flir(dataset_valid, model)
    #        net_logger.info('\n{}'.format(summarize))
    #    elif parser.dataset == 'thermal':
    #        summarize = coco_evaluate_cvidata(dataset_valid, model)
    #        net_logger.info('\n{}'.format(summarize))
    #        
    #    else:
    #        print('ERROR: Unknow dataset.')



if __name__ == '__main__':
    main()
