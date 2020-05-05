import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from architectures.utils import BBoxTransform, ClipBoxes
from architectures.utils_resnet import BasicBlock, Bottleneck
from architectures.anchors import Anchors
from architectures import losses

from architectures.backbone import shufflenetv2, densenet, mnasnet, mobilenet

import pdb


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, 
                 in_pyramid_levels=[3, 4, 5], out_pyramid_levels=[3, 4, 5, 6, 7]):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()
        self.convert_onnx = False
        self.fixed_size = None

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        if self.convert_onnx:
            print('Convert ONNX Mode at RegressoinModel')
            return out.contiguous().view(1, -1, 4)
        else:  
            # print('RegressionModel::out.shape[0] = {}'.format(str(out.shape[0])))
            return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.convert_onnx = False
        self.fixed_size = None

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        if self.convert_onnx:
            assert self.fixed_size is not None, 'ClassificationModel::fixed_size Must to be not None.'
            print('Convert ONNX Mode at ClassificationModel')
            print('ClassificationModel::fixed_size = {}'.format(self.fixed_size))
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        if self.convert_onnx:
            batch_size = 1
            width, height = self.fixed_size
            channels = self.num_classes * self.num_anchors
            # print('[b, w, h, c] = [{}, {}, {}, {}]'.format(batch_size, width, height, channels))
        else:
            # print('ClassificationModel::out1.shape = {}'.format(str(out1.shape)))
            batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        if self.convert_onnx:
            return out2.contiguous().view(1, -1, self.num_classes)
        else:
            # print('ClassificationModel::x.shape[0] = {}'.format(str(x.shape[0])))
            return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class KSevenDet(nn.Module):
    def __init__(self, cfg, iou_threshold=0.5, **kwargs):
        super(KSevenDet, self).__init__()
        # self.backbone = _get_backbone(cfg)
        self.num_classes = cfg['num_classes']
        self.feature_pyramid_levels = cfg['feature_pyramid_levels']
        self.head_pyramid_levels    = cfg['head_pyramid_levels']
        self.iou_threshold = iou_threshold

        self.convert_onnx = False

        if cfg['backbone'] == 'resnet':
            assert 0, 'not support now'
        elif cfg['backbone'] == 'efficientnet':
            assert 0, 'not support now'
        elif cfg['backbone'] == 'mobilenet':
            self.backbone = mobilenet.mobilenet_v2(feature_pyramid_level=self.feature_pyramid_levels)
        elif cfg['backbone'] == 'shuflenetv2':
            # self.backbone = shufflenetv2.shufflenet_v2_x0_5(feature_pyramid_level=self.feature_pyramid_levels)
            self.backbone = shufflenetv2.shufflenet_v2_x1_0(feature_pyramid_level=self.feature_pyramid_levels)
        elif cfg['backbone'] == 'densenet':
            self.backbone = densenet.densenet121(feature_pyramid_level=self.feature_pyramid_levels)
        elif cfg['backbone'] == 'mnasnet':
            # self.backbone = mnasnet.mnasnet0_5(feature_pyramid_level=self.feature_pyramid_levels)
            self.backbone = mnasnet.mnasnet0_75(feature_pyramid_level=self.feature_pyramid_levels)
            # self.backbone = mnasnet.mnasnet1_0(feature_pyramid_level=self.feature_pyramid_levels)
            # self.backbone = mnasnet.mnasnet1_3(feature_pyramid_level=self.feature_pyramid_levels)
        else:
            raise ValueError('Unknown backbone.')
        
        _feature_maps_channels = self.backbone.feature_maps_channels
        print(f'Feature Maps Channels : {str(_feature_maps_channels)}')
        
        # my_pyramid_levels = [3, 4, 5, 6, 7]
        # self.head = _get_head(cfg)
        if cfg['neck'] == 'fpn':
            self.neck = PyramidFeatures(*_feature_maps_channels, 
                                        in_pyramid_levels=self.feature_pyramid_levels, 
                                        out_pyramid_levels=self.head_pyramid_levels)
        else:
            raise ValueError('Unknown head.')


        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=self.num_classes)

        my_pyramid_levels = self.head_pyramid_levels
        # my_sizes   = [int(2 ** (x + 1) * 1.25) for x in my_pyramid_levels]
        my_sizes   = [int(2 ** (x + 2)) for x in my_pyramid_levels]
        my_ratios  = [1, 1.5, 2]
        #my_ratios  = [0.45, 1, 3]
        # self.anchors = Anchors(pyramid_levels=my_pyramid_levels,
        #                        sizes=my_sizes,
        #                        ratios=my_ratios,
        #                        **kwargs)
        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        # my_positive_threshold = 0.45
        # my_negative_threshold = 0.35
        # self.focalLoss = losses.FocalLoss(positive_threshold=my_positive_threshold, 
        #                                   negative_threshold=my_negative_threshold)
        self.focalLoss = losses.FocalLoss()

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, img_batch, annotations=None, return_head=False, return_loss=True):
        #if self.training:
        #    img_batch, annotations = inputs
        #else:
        #    img_batch = inputs

        features = self.backbone(img_batch)

        neck_features = self.neck(features)

        regression = torch.cat([self.regressionModel(_feature) for _feature in neck_features], dim=1)

        classification = torch.cat([self.classificationModel(_feature) for _feature in neck_features], dim=1)

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
                print('No boxes to NMS')
                # no boxes to NMS, just return
                # return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
                return [torch.zeros([1]).cuda(0), torch.zeros([1]).cuda(0), torch.zeros([1, 4]).cuda(0)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0,:,:], scores[0,:,0], 0.5)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

            return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]