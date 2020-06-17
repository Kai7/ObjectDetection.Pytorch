import onnx
import onnxruntime
import caffe2.python.onnx.backend as backend
import argparse
import yaml

import collections
import numpy as np
import torch
from torchvision import transforms
from architectures import retinanet, efficientdet
from ksevendet.architecture import ksevendet
from datasettool.dataloader import CocoDataset, FLIRDataset, CVIDataset,CSVDataset, collater, \
                                   Resizer, AspectRatioBasedSampler, Augmenter, \
                                   Normalizer, K7ImagePreprocessor
from torch.utils.data import DataLoader
from datasettool import coco_eval
from datasettool.flir_eval import evaluate_flir
from datasettool import csv_eval

from utils.tensor_compare import TensorCompare

# from torchvision.ops import nms
import skimage.draw
import skimage.io
import skimage.color
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt 

import os
import sys
import logging
LOGGING_FORMAT = '%(levelname)s:    %(message)s'

assert torch.__version__.split('.')[0] == '1'

import pdb

COLOR_LABEL = ['yellow', 'green']

def get_args():
    parser = argparse.ArgumentParser(description='KSeven ONNX inference debug tool.')
    parser.add_argument('--onnx', type=str,
                        help='Onnx file path.')
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
    parser.add_argument('--sample_path', type=str,
                        help='Demo images directory path.')
    parser.add_argument('--resize_mode', default=1, type=int,
                        help='The resize mode for Resizer')
    parser.add_argument('--threshold', default=0.6, type=float,
                        help='Object pasitive threshold.')
    parser.add_argument("--pytorch_inference", default=False, action="store_true" , 
                        help='Use pytorch to inference.')
    parser.add_argument("--onnx_inference", default=False, action="store_true" , 
                        help='Use onnx to inference.')
    parser.add_argument("--compare_head_tensor", default=False, action="store_true" , 
                        help='Compare head tensor between pytorch and onnx.')
    parser.add_argument("--dump_sample_npz", default=False, action="store_true" , 
                        help='Whether dump sample image npz file.')
    parser.add_argument("--log", default=False, action="store_true" , 
                        help="Write log file.")

    args = parser.parse_args()
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_cfg = yaml.safe_load(f)
        setattr(args, 'architecture', model_cfg.pop('architecture'))
        setattr(args, 'model_cfg', model_cfg)

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

def main():
    #args = parser.parse_args()
    args = get_args()

    if args.pytorch_inference:
        assert args.architecture, 'Must provide --architecture when pytorch inference.'
        assert args.resume, 'Must provide --resume when pytorch inference.'
    if args.onnx_inference:
        assert args.onnx, 'Must provide --onnx when onnx inference.'

    #transform = transforms.Compose([Normalizer(inference_mode=True), 
    #                                Resizer(height=input_height, width=input_width, inference_mode=True)])
    #my_img_preprocessor = K7ImagePreprocessor(
    #                          mean = np.array([[[0.485, 0.456, 0.406]]]),
    #                          std  = np.array([[[0.229, 0.224, 0.225]]]),
    #                          resize_height=input_height, resize_width=input_width)
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

    net_logger = get_logger(name='ONNX Inference Logger', args=args)
    in_h, in_w = tuple(map(int, args.input_shape.split(',')))
    #in_h, in_w = int(in_h), int(in_w)
    net_logger.info(f'Input Tensor Size: [ {in_h}, {in_w}]')
    net_logger.info('Positive Threshold: {:.2f}'.format(args.threshold))

    _normalizer = Normalizer(inference_mode=True)
    if args.resize_mode == 0:
        assert 0, 'not use...'
        #_resizer = Resizer(min_side=_shape_1, max_side=_shape_2, resize_mode=args.resize_mode, logger=net_logger, inference_mode=True)
    elif args.resize_mode == 1:
        _resizer = Resizer(height=in_h, width=in_w, resize_mode=args.resize_mode, logger=net_logger, inference_mode=True)
    else:
        raise ValueError('Illegal resize mode.')

    transfrom_funcs_valid = [
        _normalizer,
        _resizer,
    ]
    transform = transforms.Compose(transfrom_funcs_valid)

    pytorch_model = None
    if args.pytorch_inference:
        net_logger.info('Build pytorch model...')
        build_param = {'logger': net_logger}
        if args.architecture == 'ksevendet':
            pytorch_model = ksevendet.KSevenDet(ksevendet_cfg, num_classes=args.num_classes, pretrained=False, **build_param)
        elif args.architecture == 'retinanet-p45p6':
            pytorch_model = retinanet.retinanet_p45p6(num_classes=args.num_classes, **build_param)
        elif args.architecture.split('-')[0] == 'retinanet':
            pytorch_model = retinanet.build_retinanet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
        elif args.architecture.split('-')[0] == 'efficientdet':
            pytorch_model = efficientdet.build_efficientdet(args.architecture, num_classes=args.num_classes, pretrained=False, **build_param)
        else:
            assert 0, 'architecture error'
    
        if args.resume is not None:
            #net_logger.info('Loading Checkpoint : {}'.format(args.resume))
            #model.load_state_dict(torch.load(args.resume))
            net_logger.info('Loading Weights from Checkpoint : {}'.format(args.resume))
            try:
                ret = pytorch_model.load_state_dict(torch.load(args.resume), strict=False)
            except RuntimeError as e:
                net_logger.warning(f'Ignoring {e}')
                net_logger.warning(f'Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')
        else:
            raise ValueError('Must provide --resume when testing.')

        use_gpu = True

        if use_gpu:
            if torch.cuda.is_available():
                pytorch_model = pytorch_model.cuda()

        #if torch.cuda.is_available():
        #   pytorch_model = torch.nn.DataParallel(pytorch_model).cuda()
        #else:
        #   pytorch_model = torch.nn.DataParallel(pytorch_model)

        pytorch_model.eval()
        net_logger.info('Initialize Pytorch Model...  Finished')

    onnx_model = None
    if args.onnx_inference:
        net_logger.info('Build onnx model...')
        net_logger.info('Onnx loading...')
        # Load the ONNX model
        #model = onnx.load('./retinanet-tiny.onnx')
        onnx_model = onnx.load(args.onnx)
        print(type(onnx_model))

        # Check that the IR is well formed
        net_logger.info('Onnx checking...')
        onnx.checker.check_model(onnx_model)

        ort_session = onnxruntime.InferenceSession(args.onnx)
        net_logger.info('Onnx initialize... Done')

        ## Print a human readable representation of the graph
        #onnx.helper.printable_graph(onnx_model.graph)

        #rep = backend.prepare(onnx_model, device="CUDA:0") # or "CPU"
        ## For the Caffe2 backend:
        ##     rep.predict_net is the Caffe2 protobuf for the network
        ##     rep.workspace is the Caffe2 workspace for the network
        ##       (see the class caffe2.python.onnx.backend.Workspace)


    sample_files = os.listdir(args.sample_path)
    sample_files.sort()

    tensor_comp = TensorCompare()

    # for f in sample_files[:1]:
    for f in sample_files:
        if f.split('.')[-1] not in ['png', 'jpg']:
            continue
        print(f'loading image... {f}')

        img = Image.open(os.path.join(args.sample_path, f)).convert('RGB')
        a_img = np.array(img)
        a_img = a_img.astype(np.float32) / 255.0

        #t_img = transform(a_img)
        #t_img = torch.unsqueeze(t_img, 0)
        #t_img = t_img.permute(0, 3, 1, 2)

        a_img = transform(a_img).numpy()
        # a_img = my_img_preprocessor(a_img)
        a_img = np.expand_dims(a_img, axis=0)
        a_img = a_img.transpose((0, 3, 1, 2))

        if args.dump_sample_npz:
            _npz_name = os.path.join(args.sample_path, f.split('.')[0]+'_preprocess.npz')
            print(f'save npz to ... {_npz_name}')
            np.savez(_npz_name, input=a_img)

        #tt = tensor_comp.compare(to_numpy(t_img), a_img)
        #pdb.set_trace()

        net_logger.info('pytorch inference start ...')
        # scores, labels, boxes = pytorch_model(aa_img)
        pytorch_scores, pytorch_labels, pytorch_boxes = pytorch_model(torch.from_numpy(a_img).to(torch.device("cuda:0")))
        net_logger.info('inference done.')
        if args.compare_head_tensor:
            pytorch_classification, pytorch_regression = pytorch_model(torch.from_numpy(a_img).to(torch.device("cuda:0")), return_head=True)

        pytorch_scores = pytorch_scores.cpu()
        pytorch_labels = pytorch_labels.cpu()
        pytorch_boxes  = pytorch_boxes.cpu()

        # change to (x, y, w, h) (MS COCO standard)
        pytorch_boxes[:, 2] -= pytorch_boxes[:, 0]
        pytorch_boxes[:, 3] -= pytorch_boxes[:, 1]

        draw = ImageDraw.Draw(img)
        for box_id in range(pytorch_boxes.shape[0]):
            score = float(pytorch_scores[box_id])
            label = int(pytorch_labels[box_id])
            box = pytorch_boxes[box_id, :]

            # scores are sorted, so we can break
            if score < args.threshold:
                break

            x, y, w, h = box
            # draw.rectangle(tuple([x, y, x+w, y+h]), width = 2, outline =COLOR_LABEL[label])
            
        _img_save_path = os.path.join(args.sample_path, os.path.basename(f)[:-4]+'_inference_visual.jpg')

        #onnx_input = a_img
        #onnx_input = to_tensor(a_img)
        #onnx_input.unsqueeze_(0)
        print('Onnx Inference')
        print('Input.shape = {}'.format(str(a_img.shape)))
        print('start inference')
        # outputs = rep.run(a_img)
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(onnx_input)}
        ort_inputs = {ort_session.get_inputs()[0].name: a_img}

        net_logger.info('pytorch inference start ...')
        ort_outs = ort_session.run(None, ort_inputs)
        net_logger.info('inference done.')

        # onnx_out = ort_outs[0]
        # To run networks with more than one input, pass a tuple
        # rather than a single numpy ndarray.

        #print(f'len(ort_outs) = {len(ort_outs)}')
        #for i, t in enumerate(ort_outs):
        #    print(f'out[{i}] type is {type(t)}')
        #    print(f'out[{i}] shape is {t.shape}')
        
        classification, regression = ort_outs

        # this is wrong code?
        anchors = pytorch_model.anchors(a_img)
        # print(anchors)
        # print(anchors.shape)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        transformed_anchors = pytorch_model.regressBoxes(anchors, torch.from_numpy(regression).float().to(device))
        transformed_anchors = pytorch_model.clipBoxes(transformed_anchors, a_img)
        transformed_anchors = to_numpy(transformed_anchors)

        #scores = torch.max(classification, dim=2, keepdim=True)[0]
        onnx_scores = np.max(classification, axis=2, keepdims=True)[0]

        # scores_over_thresh = (onnx_scores > 0.05)[0, :, 0]
        onnx_scores_over_thresh = (onnx_scores > 0.05)[:, 0]

        # print(f'onnx_scores_over_thresh.sum() = {onnx_scores_over_thresh.sum()}')
        if onnx_scores_over_thresh.sum() == 0:
            print('No boxes to NMS')
            # no boxes to NMS, just return
            # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, onnx_scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, onnx_scores_over_thresh, :]
        #onnx_scores = onnx_scores[:, onnx_scores_over_thresh, :]
        onnx_scores = onnx_scores[onnx_scores_over_thresh, :]

        anchors_nms_idx = nms(transformed_anchors[0,:,:], onnx_scores[:,0], 0.5)
        # pdb.set_trace()
        onnx_nms_scores = classification[0, anchors_nms_idx, :].max(axis=1)
        onnx_nms_class  = classification[0, anchors_nms_idx, :].argmax(axis=1)
        onnx_boxes      = transformed_anchors[0, anchors_nms_idx, :]
        onnx_boxes[:, 2] -= onnx_boxes[:, 0]
        onnx_boxes[:, 3] -= onnx_boxes[:, 1]


        if args.compare_head_tensor:
            onnx_classification, onnx_regression = ort_outs
            net_logger.info('compare classification ...')
            _result = tensor_comp.compare(to_numpy(pytorch_classification), onnx_classification)
            print(f'Pass {_result[0]}')
            for k in _result[2]:
                print('{:>20} : {:<2.8f}'.format(k, _result[2][k]))
                if k == 'close_order':
                    print(_result)
            if not _result[0]:
                print('Not similar.')
                exit(0)
            net_logger.info('compare regression ...')
            _result = tensor_comp.compare(to_numpy(pytorch_regression), onnx_regression)
            print(f'Pass {_result[0]}')
            for k in _result[2]:
                print('{:>20} : {:<2.8f}'.format(k, _result[2][k]))

        
        #comp_idx = min(len(onnx_nms_scores), len(pytorch_scores))
        #print(onnx_nms_scores[:comp_idx])
        #print(to_numpy(pytorch_scores[:comp_idx]))

        #print(onnx_nms_class[:comp_idx])
        #print(to_numpy(pytorch_labels[:comp_idx]))

        #print(onnx_boxes[:comp_idx])
        #print(to_numpy(pytorch_boxes[:comp_idx]))

        #print(len(outputs))
        #print(outputs[0])
        #print('Output.shape = {}'.format(str(outputs[0].shape)))
        #print(type(outputs))

def nms(boxes, scores, iou_thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        # print(i)
        # pdb.set_trace()
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# pdb.set_trace()
if __name__ == '__main__':
    main()
    print('Done')