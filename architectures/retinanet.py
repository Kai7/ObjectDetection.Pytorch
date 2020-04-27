import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from architectures.utils import BBoxTransform, ClipBoxes
from architectures.utils_resnet import BasicBlock, Bottleneck
from architectures.anchors import Anchors
from architectures import losses

import pdb

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

MODEL_ZOO = 'model_zoo'

class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
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


class RetinaNet(nn.Module):
    def __init__(self, num_classes, block, layers, iou_threshold=0.5, **kwargs):
        self.inplanes = 64
        super(RetinaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.iou_threshold = iou_threshold

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        my_pyramid_levels = [3, 4, 5, 6, 7]
        # my_sizes   = [int(2 ** (x + 1) * 1.25) for x in my_pyramid_levels]
        my_sizes   = [int(2 ** (x + 2)) for x in my_pyramid_levels]
        my_ratios  = [1, 1.5, 2]
        #my_ratios  = [0.45, 1, 3]
        self.anchors = Anchors(pyramid_levels=my_pyramid_levels,
                               sizes=my_sizes,
                               ratios=my_ratios,
                               **kwargs)
        # self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        # my_positive_threshold = 0.45
        # my_negative_threshold = 0.35
        # self.focalLoss = losses.FocalLoss(positive_threshold=my_positive_threshold, 
        #                                   negative_threshold=my_negative_threshold)
        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, inputs):

        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])

        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

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

def retinanet(depth, num_classes=80, pretrained=False, **kwargs):
    assert depth in [18, 34, 50, 101, 152]
    if depth == 18:
        return retinanet18(num_classes, pretrained=pretrained, **kwargs)
    elif depth == 34:
        return retinanet34(num_classes, pretrained=pretrained, **kwargs)
    elif depth == 50:
        return retinanet50(num_classes, pretrained=pretrained, **kwargs)
    elif depth == 101:
        return retinanet101(num_classes, pretrained=pretrained, **kwargs)
    elif depth == 152:
        return retinanet152(num_classes, pretrained=pretrained, **kwargs)
    else:
        return None

def retinanet18(num_classes, pretrained=False, **kwargs):
    """Constructs a retinanet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'], model_dir=MODEL_ZOO)
        model.load_state_dict(state_dict, strict=False)
        for _l in state_dict:
            print('{:^40}{:>20}'.format(_l, str(state_dict[_l].shape)))
        # pdb.set_trace()
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def retinanet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def retinanet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def retinanet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def retinanet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model

########################################################################################################################

class PyramidFeaturesTiny(nn.Module):
    def __init__(self, C2_size, C3_size, feature_size=256):
        super(PyramidFeaturesTiny, self).__init__()
        self.convert_onnx  = False
        self.pyramid_sizes = None

        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4 = nn.Conv2d(C3_size, feature_size, kernel_size=3, stride=2, padding=1)


    def forward(self, inputs):
        if self.convert_onnx:
            assert self.pyramid_sizes is not None, 'PyramidFeatures::pyramid_sizes must be not None'
            print('Convert ONNX Mode at PyramidFeatures')
            for i in range(len(self.pyramid_sizes)):
                print('[{}] Pyramid Size = [{:>4},{:>4}]'.format(i, *self.pyramid_sizes[i]))

        C2, C3 = inputs
        P3_x = self.P3_1(C3)
        if self.convert_onnx:
            self.P3_upsampled.size = list(self.pyramid_sizes[0])
            self.P3_upsampled.scale_factor = None
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)

        P4_x = self.P4(C3)

        return [P2_x, P3_x, P4_x]

class RetinaNetTiny(nn.Module):
    def __init__(self, num_classes, block, layers, iou_threshold=0.5, **kwargs):
        self.inplanes = 64
        super(RetinaNetTiny, self).__init__()
        self.convert_onnx = False
        self.fixed_size = None

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.iou_threshold = iou_threshold

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels]
            #fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
            #             self.layer4[layers[3] - 1].conv2.out_channels]
        #elif block == Bottleneck:
        #    fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
        #                 self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeaturesTiny(fpn_sizes[0], fpn_sizes[1])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        my_pyramid_levels = [2, 3, 4]
        # my_sizes   = [int(2 ** (x + 1) * 1.25) for x in my_pyramid_levels]
        my_sizes   = [int(2 ** (x + 2)) for x in my_pyramid_levels]
        my_ratios  = [1, 1.5, 2]
        #my_ratios  = [0.45, 1, 3]
        self.anchors = Anchors(pyramid_levels=my_pyramid_levels,
                               sizes=my_sizes,
                               ratios=my_ratios,
                               **kwargs)
        # self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        # my_positive_threshold = 0.45
        # my_negative_threshold = 0.35
        # self.focalLoss = losses.FocalLoss(positive_threshold=my_positive_threshold, 
        #                                   negative_threshold=my_negative_threshold)
        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)

        my_pyramid_levels = [2, 3, 4]

        if self.convert_onnx:
            assert self.fpn.convert_onnx, 'PyramidFeature.convert_onnx must be True'
            self.fpn.pyramid_sizes = list()
            h, w = self.fixed_size
            for i in range(len(my_pyramid_levels)):
                _fixed_size = ( int(h/2**(i+2)), int(w/2**(i+2)) )
                self.fpn.pyramid_sizes.append(_fixed_size)

        features = self.fpn([x2, x3])
        # Return [P2, P3, P4]

        if self.convert_onnx:
            assert self.regressionModel.convert_onnx, 'RegressionModel.convert_onnx must be True'
        #_reg = [self.regressionModel(feature) for feature in features]
        #regression = torch.cat(_reg[:2], dim=1)
        #for i in range(2, len(_reg)):
        #    regression = torch.cat([regression, _reg[i]], dim=1)
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        if self.convert_onnx:
            assert self.fixed_size, 'RetinaNet.fixed_size must be not None'
            assert self.classificationModel.convert_onnx, 'ClassificationModel.convert_onnx must be True'
            print('Fixed Size = [{:>4},{:>4}]'.format(*self.fixed_size))
            _cls = list()
            h, w = self.fixed_size
            for i in range(len(features)):
                _fixed_size = ( int(h/2**(i+2)), int(w/2**(i+2)) )
                print('[{}] Pyramid Size : [{:>4},{:>4}]'.format(i, *_fixed_size))
                self.classificationModel.fixed_size = _fixed_size
                _cls.append(self.classificationModel(features[i]))
            #classification = torch.cat(_cls[:2], dim=1)
            #for i in range(2, len(_cls)):
            #    classification = torch.cat([classification, _cls[i]], dim=1)
            classification = torch.cat(_cls, dim=1)
        else:
            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if self.training:
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


def retinanet_tiny(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNetTiny(num_classes, BasicBlock, [2, 2, 2], **kwargs)
    return model

#############################################################################################################

class PyramidFeaturesP45P6(nn.Module):
    def __init__(self, C4_size, C5_size, feature_size=256):
        super(PyramidFeaturesP45P6, self).__init__()
        self.convert_onnx  = False
        self.pyramid_sizes = None


        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=5, stride=3, padding=1)

    def forward(self, inputs):
        if self.convert_onnx:
            assert self.pyramid_sizes is not None, 'PyramidFeatures::pyramid_sizes must be not None'
            print('Convert ONNX Mode at PyramidFeatures')
            for i in range(len(self.pyramid_sizes)):
                print('[{}] Pyramid Size = [{:>4},{:>4}]'.format(i, *self.pyramid_sizes[i]))

        C4, C5 = inputs
        P5_x = self.P5_1(C5)
        if self.convert_onnx:
            self.P5_upsampled.size = list(self.pyramid_sizes[0])
            self.P5_upsampled.scale_factor = None
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        P4_x = self.P4_2(P4_x)

        P6_x = self.P6(C5)

        return [P4_x, P5_x, P6_x]


class RetinaNetP45P6(nn.Module):
    def __init__(self, num_classes, block, layers, iou_threshold=0.5, 
                 pyramid_levels=[4,5,6], **kwargs):
        self.inplanes = 64
        super(RetinaNetP45P6, self).__init__()
        self.convert_onnx = False
        self.fixed_size = None

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.iou_threshold = iou_threshold
        self.pyramid_levels = pyramid_levels

        if block == BasicBlock:
            fpn_sizes = [self.layer3[layers[1] - 1].conv2.out_channels, self.layer4[layers[2] - 1].conv2.out_channels]
            print(f'fpn_sizes = {str(fpn_sizes)}')
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeaturesP45P6(fpn_sizes[0], fpn_sizes[1])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        # my_pyramid_levels = [4, 5, 6]
        # my_sizes   = [int(2 ** (x + 1) * 1.25) for x in my_pyramid_levels]
        my_strides = [2**4, 2**5, 3*(2**5)]
        my_sizes   = [2**5, 2**6, int(2.5*(2**6))]
        print(f'my_stride = {str(my_strides)}')
        print(f'my_sizes = {str(my_sizes)}')
        my_ratios  = [1, 1.5, 2]
        #my_ratios  = [0.45, 1, 3]
        self.anchors = Anchors(pyramid_levels=self.pyramid_levels,
                               strides=my_strides,
                               sizes=my_sizes,
                               ratios=my_ratios,
                               **kwargs)
        # self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        # my_positive_threshold = 0.45
        # my_negative_threshold = 0.35
        # self.focalLoss = losses.FocalLoss(positive_threshold=my_positive_threshold, 
        #                                   negative_threshold=my_negative_threshold)
        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        print(f'_make_layer, inplanes = {self.inplanes}')
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

        x = self.conv1(img_batch)   # c1
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)         # c2

        x1 = self.layer1(x)                 
        x2 = self.layer2(x1)        # c3
        x3 = self.layer3(x2)        # c4
        x4 = self.layer4(x3)        # c5

        my_pyramid_levels = [4, 5, 6]
        my_strides = my_strides = [2**4, 2**5, 3*(2**5)]

        if self.convert_onnx:
            assert self.fpn.convert_onnx, 'PyramidFeature.convert_onnx must be True'
            self.fpn.pyramid_sizes = list()
            h, w = self.fixed_size
            for s in my_strides:
                _fixed_size = ( int(h/s), int(w/s) ) 
                self.fpn.pyramid_sizes.append(_fixed_size)

        features = self.fpn([x3, x4])
        # Return [P2, P3, P4]

        if self.convert_onnx:
            assert self.regressionModel.convert_onnx, 'RegressionModel.convert_onnx must be True'
        #_reg = [self.regressionModel(feature) for feature in features]
        #regression = torch.cat(_reg[:2], dim=1)
        #for i in range(2, len(_reg)):
        #    regression = torch.cat([regression, _reg[i]], dim=1)
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        # print(regression.shape)

        if self.convert_onnx:
            assert self.fixed_size, 'RetinaNet.fixed_size must be not None'
            assert self.classificationModel.convert_onnx, 'ClassificationModel.convert_onnx must be True'
            print('Fixed Size = [{:>4},{:>4}]'.format(*self.fixed_size))
            _cls = list()
            h, w = self.fixed_size
            # for i in range(len(features)):
            for i, s in enumerate(my_strides):
                _fixed_size = ( int(h/s), int(w/s) ) 
                print('[{}] Pyramid Size : [{:>4},{:>4}]'.format(i, *_fixed_size))
                self.classificationModel.fixed_size = _fixed_size
                _cls.append(self.classificationModel(features[i]))
            #classification = torch.cat(_cls[:2], dim=1)
            #for i in range(2, len(_cls)):
            #    classification = torch.cat([classification, _cls[i]], dim=1)
            classification = torch.cat(_cls, dim=1)
        else:
            classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
            # print(classification.shape)

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


def retinanet_p45p6(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaNetP45P6(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    return model