import torch.nn as nn
import torch
import numpy as np
import math
from torchvision.ops import nms

from ksevendet.architecture.utils import BBoxTransform, ClipBoxes
from ksevendet.architecture.anchors import Anchors
from ksevendet.architecture.losses import FocalLoss

# from architectures.backbone import shufflenetv2, densenet, mnasnet, mobilenet
from ksevendet.architecture.backbone import shufflenetv2, mobilenetv2, mobilenetv3, efficientnet, resnet, densenet, res2net, senet, sknet
from ksevendet.architecture.neck import FPN, PANetFPN, BiFPN, TFDBiFPN, BiFPNORG
from ksevendet.architecture.head import Regressor, Classifier, RegressorV1, RegressorV2, ClassifierV1, ClassifierV2

import ksevendet.architecture.backbone.registry as registry

import pdb

class KSevenDet(nn.Module):
    def __init__(self, cfg, num_classes=80, iou_threshold=0.5, **kwargs):
        super(KSevenDet, self).__init__()
        self.convert_onnx = False
        self.fixed_size   = None

        logger = kwargs.get('logger', None)
        # self.backbone = _get_backbone(cfg)
        self.num_classes = num_classes
        self.backbone_feature_pyramid_levels = cfg['backbone_feature_pyramid_levels']
        self.neck_feature_pyramid_levels     = cfg['neck_feature_pyramid_levels']
        self.head_feature_pyramid_levels     = cfg['head_feature_pyramid_levels']
        self.build_config = cfg
        self.iou_threshold = iou_threshold

        self.backbone = self._build_backbone(cfg, logger=logger)
        
        _feature_channels = self.backbone.feature_channels()
        if logger:
            logger.info(f'Backbone Features Num : {str(_feature_channels)}')
        
        # my_pyramid_levels = [3, 4, 5, 6, 7]
        # self.head = self._build_head(cfg)
        # fpn_features_num = cfg.get('fpn_features_num', 256)
        fpn_features_num = cfg['neck_config']['features_num']
        if logger:
            logger.info(f'FPN Features Num : {fpn_features_num}')
        
        if cfg['neck'] == 'fpn':
            # self.neck = FPN(*_feature_channels, 
            #                 in_pyramid_levels=self.backbone_feature_pyramid_levels, 
            #                 out_pyramid_levels=self.neck_feature_pyramid_levels)
            self.neck = PANetFPN(_feature_channels, 
                                 in_pyramid_levels=self.backbone_feature_pyramid_levels, 
                                 out_pyramid_levels=self.neck_feature_pyramid_levels,
                                 features_num=fpn_features_num,
                                 logger=logger)
        elif cfg['neck'] == 'panet-fpn':
            self.neck = PANetFPN(_feature_channels, 
                                 in_pyramid_levels=self.backbone_feature_pyramid_levels, 
                                 out_pyramid_levels=self.neck_feature_pyramid_levels,
                                 features_num=fpn_features_num,
                                 panet_buttomup=True,
                                 logger=logger)
        elif cfg['neck'] == 'bifpn':
            # bifpn_repeats = cfg.get('bifpn_repeats', 2)
            # bifpn_attention = cfg.get('bifpn_attention', False)
            bifpn_repeats = cfg['neck_config']['bifpn_repeats']
            bifpn_attention = cfg['neck_config']['bifpn_attention']
            #_bifpn_modules = [BiFPN(_feature_channels, 
            #                        in_pyramid_levels=self.backbone_feature_pyramid_levels, 
            #                        neck_pyramid_levels=self.neck_feature_pyramid_levels,
            #                        out_pyramid_levels=self.head_feature_pyramid_levels,
            #                        features_num=fpn_features_num,
            #                        first_time=True if _ == 0 else False,
            #                        attention=bifpn_attention,
            #                        logger=logger)
            #                        for _ in range(bifpn_repeats)]
            #self.neck = nn.Sequential(*_bifpn_modules)
            self.neck = BiFPN(_feature_channels, 
                              in_pyramid_levels=self.backbone_feature_pyramid_levels, 
                              neck_pyramid_levels=self.neck_feature_pyramid_levels,
                              out_pyramid_levels=self.head_feature_pyramid_levels,
                              features_num=fpn_features_num,
                              repeats=bifpn_repeats,
                              attention=bifpn_attention,
                              logger=logger)
        elif cfg['neck'] == 'tfdbifpn':
            # bifpn_repeats = cfg.get('bifpn_repeats', 2)
            # bifpn_attention = cfg.get('bifpn_attention', False)
            bifpn_repeats = cfg['neck_config']['bifpn_repeats']
            bifpn_attention = cfg['neck_config']['bifpn_attention']
            if logger:
                logger.info(f'==== Build Neck Layer (TFD) ====================')
                logger.info(f'BiFPN Repeats: {bifpn_repeats}')
            _bifpn_modules = [TFDBiFPN(_feature_channels, 
                                       in_pyramid_levels=self.backbone_feature_pyramid_levels, 
                                       neck_pyramid_levels=self.neck_feature_pyramid_levels,
                                       out_pyramid_levels=self.head_feature_pyramid_levels,
                                       features_num=fpn_features_num,
                                       first_time=True if _ == 0 else False,
                                       attention=bifpn_attention,
                                       logger=logger)
                                       for _ in range(bifpn_repeats)]
            self.neck = nn.Sequential(*_bifpn_modules)
        else:
            raise ValueError(f'Unknown neck {cfg["neck"]}')

        # my_pyramid_levels = [3, 4, 5, 6, 7]
        my_pyramid_levels = self.head_feature_pyramid_levels
        my_strides = [2 ** x for x in my_pyramid_levels]
        # my_sizes   = [2 ** (x + 2) for x in my_pyramid_levels]
        #my_sizes   = [2 ** (x + 1) * 1.5 for x in my_pyramid_levels]
        my_sizes   = [2 ** (x + 1) * 1.25 for x in my_pyramid_levels]
        # my_ratios  = np.array([0.5, 1, 2])  # shape_0 / shape_1
        my_ratios  = np.array([1, 2])  # shape_0 / shape_1
        my_scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        my_num_anchors = len(my_ratios) * len(my_scales)

        # my_sizes   = [int(2 ** (x + 1) * 1.25) for x in my_pyramid_levels]
        # my_sizes   = [int(2 ** (x + 2)) for x in my_pyramid_levels]
        # my_ratios  = [1, 1.5, 2]
        # my_ratios  = [0.45, 1, 3]
        #self.anchors = Anchors(pyramid_levels=my_pyramid_levels,
        #                       strides=my_strides,
        #                       sizes=my_sizes,
        #                       ratios=my_ratios,
        #                       scales=my_scales,
        #                       **kwargs)
        self.anchors = Anchors(pyramid_levels=self.head_feature_pyramid_levels, logger=logger, 
                               **cfg['anchors_config'])
        # self.anchors = Anchors()

        self.head_version = 1
        self.regressor  = Regressor(fpn_features_num, num_anchors=my_num_anchors, 
                                    num_pyramid_levels=len(self.head_feature_pyramid_levels), logger=logger, **cfg['head_config'])
        self.classifier = Classifier(fpn_features_num, num_anchors=my_num_anchors, num_classes=self.num_classes, 
                                     num_pyramid_levels=len(self.head_feature_pyramid_levels), logger=logger, **cfg['head_config'])


        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        # my_positive_threshold = 0.45
        # my_negative_threshold = 0.35
        # self.focalLoss = losses.FocalLoss(positive_threshold=my_positive_threshold, 
        #                                   negative_threshold=my_negative_threshold)
        self.focalLoss = FocalLoss()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()


        self.freeze_bn()

    def _build_backbone(self, cfg, **kwargs):
        logger = kwargs.get('logger', None)
        # print(registry._model_to_module.keys())
        assert cfg['backbone'] in registry._module_to_models, f'Backbone: {cfg["backbone"]} not support.'

        backbone_build_info = {
            'features_only': True, 
            'backbone_feature_pyramid_level': self.backbone_feature_pyramid_levels,
        }
        if cfg.get('variant', None):
            assert registry.is_model(cfg['variant']), f'Variant: {cfg["variant"]} not support.'
            build_fn = registry.model_entrypoint(cfg['variant'])
            if logger:
                logger.info(f'Use default variant: {build_fn}')
            return build_fn(**backbone_build_info, **kwargs)
        else:
            assert cfg.get('backbone_config')

            print('Not support now.')
            exit(0)

        # if cfg['backbone'] == 'resnet':
        #     self.backbone = resnet.resnet18(**backbone_build_info)
        #     self.backbone = resnet.resnet34(**backbone_build_info)
        # elif cfg['backbone'] == 'res2net':
        #     self.backbone = res2net.res2net50_14w_8s(**backbone_build_info)
        #     self.backbone = res2net.res2net50_26w_4s(**backbone_build_info)
        #     self.backbone = res2net.res2net50_26w_6s(**backbone_build_info)
        #     self.backbone = res2net.res2net50_26w_8s(**backbone_build_info)
        #     self.backbone = res2net.res2net50_48w_2s(**backbone_build_info)
        #     self.backbone = res2net.res2net101_26w_4s(**backbone_build_info)
        #     self.backbone = res2net.res2next50(**backbone_build_info)
        # if cfg['backbone'] == 'sknet':
        #     self.backbone = sknet.skresnet18(**backbone_build_info)
        #     self.backbone = sknet.skresnet34(**backbone_build_info)
        #     self.backbone = sknet.skresnet50(**backbone_build_info)
        #     self.backbone = sknet.skresnet50d(**backbone_build_info)
        # elif cfg['backbone'] == 'efficientnet':
        #     assert 0, 'not support now'
        # elif cfg['backbone'] == 'mobilenetv2':
        #     self.backbone = mobilenetv2.mobilenetv2(**backbone_build_info)
        # elif cfg['backbone'] == 'shufflenetv2':
        #     self.backbone = shufflenetv2.shufflenetv2_x0_5(f**backbone_build_info)
        #     self.backbone = shufflenetv2.shufflenetv2_x1_0(**backbone_build_info)
        # elif cfg['backbone'] == 'densenet':
        #     self.backbone = densenet.densenet121(**backbone_build_info)
        # elif cfg['backbone'] == 'mnasnet':
        #     self.backbone = mnasnet.mnasnet0_5(**backbone_build_info)
        #     self.backbone = mnasnet.mnasnet0_75(**backbone_build_info)
        #     self.backbone = mnasnet.mnasnet1_0(**backbone_build_info)
        #     self.backbone = mnasnet.mnasnet1_3(**backbone_build_info)
        # else:
        #     raise ValueError('Unknown backbone.')


    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, img_batch, annotations=None, return_head=False, return_loss=False):
        #if self.training:
        #    img_batch, annotations = inputs
        #else:
        #    img_batch = inputs

        features = self.backbone(img_batch)


        #print(self.anchors.strides)
        #print(self.neck.pyramid_sizes)
        #exit(0)

        neck_features = self.neck(features)

        # for x_feature in neck_features:
        #     print(x_feature.shape)
        # exit(0)

        if self.convert_onnx:
            assert self.classifier.convert_onnx, 'HEAD-Classifier::convert_onnx must be True'
            assert self.regressor.convert_onnx, 'HEAD-Regressor::convert_onnx must be True'
            self.classifier.pyramid_sizes = list()
            h, w = self.fixed_size
            for s in self.anchors.strides:
                _fixed_size = ( int(h/s), int(w/s) ) 
                self.classifier.pyramid_sizes.append(_fixed_size)

        if self.head_version == 1:
            regression = self.regressor(neck_features)
            classification = self.classifier(neck_features)
        else:
            regression = torch.cat([self.regressor(_feature) for _feature in neck_features], dim=1)
            classification = torch.cat([self.classifier(_feature) for _feature in neck_features], dim=1)

        # print(f'Regression Shape     : {str(regression.shape)}')
        # print(f'Classification Shape : {str(classification.shape)}')

        if return_head:
            return [classification, regression]

        anchors = self.anchors(img_batch)
        # print(f'Image Shape: {str(img_batch.shape)}')
        # print(f'Anchor Shape: {str(anchors.shape)}')

        #if self.training:
        if return_loss:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            if self.convert_onnx:
                print('Ignore post-process')
                return [classification, regression]

            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            #if self.convert_onnx:
            #    print('Ignore post-process')
            #    return [classification, transformed_anchors]

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.05)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                # print('No boxes to NMS')
                # no boxes to NMS, just return
                # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
                return [torch.zeros([1]).cuda(0), torch.zeros([1]).cuda(0), torch.zeros([1, 4]).cuda(0)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]

    def set_onnx_convert_info(self, fixed_size=(512, 512)):
        self.fixed_size = fixed_size 
        self.convert_onnx = True

        pyramid_sizes = list()
        h, w = self.fixed_size
        for s in self.anchors.strides:
            _fixed_size = ( int(h/s), int(w/s) ) 
            pyramid_sizes.append(_fixed_size)
        #if self.build_config['neck'] == 'bifpn':
        #    for m in self.neck:
        #        m.convert_onnx = True
        #        m.pyramid_sizes = pyramid_sizes
        #else:
        #    self.neck.convert_onnx = True
        #    self.neck.pyramid_sizes = pyramid_sizes
        self.neck.set_onnx_convert_info(pyramid_sizes)
        self.regressor.convert_onnx = True
        self.classifier.convert_onnx = True
