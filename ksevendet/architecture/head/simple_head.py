import torch
import torch.nn as nn
import math


__all__ = ['Regressor', 'Classifier', 'RegressorV1', 'ClassifierV1', 'RegressorV2', 'ClassifierV2']

ONNX_EXPORT = True

'''
Source from: zylo117 / Yet-Another-EfficientDet-Pytorch 
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

'''
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1, 
                 norm=True, activation=True, act_func=Swish, onnx_export=ONNX_EXPORT):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                        bias=False, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class Regressor(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_features_num, num_anchors=9, 
                 features_num=256, layers_num=3, num_pyramid_levels=5, head_type='simple', act_type='relu', onnx_export=ONNX_EXPORT,
                 **kwargs):
        assert head_type in ['simple', 'efficient']
        assert act_type in ['relu', 'swish']
        super(Regressor, self).__init__()
        self.layers_num = layers_num
        self.num_pyramid_levels = num_pyramid_levels
        self.head_type = head_type

        logger = kwargs.get('logger', None)
        if logger:
            logger.info(f'==== Build Head Layer ====================')
            logger.info(f'Head Type    : Regression ({head_type} + {act_type})')
            logger.info(f'Features Num : {features_num}')
            logger.info(f'Anchors Num  : {num_anchors}')
            logger.info(f'Layers Num   : {layers_num}')

        _conv_block = SeparableConvBlock if head_type == 'efficient' else nn.Conv2d
        _conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if head_type == 'efficient':
            _conv_kwargs.update({'norm': False, 'activation': False})
        self.conv_list = nn.ModuleList(
            [_conv_block(in_features_num if i == 0 else features_num, 
                         features_num, **_conv_kwargs) for i in range(layers_num)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(features_num, 
                                           momentum=0.01, eps=1e-3) for i in range(layers_num)]) for j in
             range(num_pyramid_levels)])

        self.header = _conv_block(features_num, num_anchors * 4, **_conv_kwargs)
        
        if act_type == 'swish':
            self.act_fn = MemoryEfficientSwish() if not onnx_export else Swish()
        else:
            self.act_fn = nn.ReLU()

        self._initialize_weights(logger=logger)

    def _initialize_weights(self, logger=None):
        if logger:
            logger.info('Initializing weights for Regressor...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.layers_num), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.act_fn(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class Classifier(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_features_num, num_anchors=9, num_classes=80, 
                 features_num=256, layers_num=3, num_pyramid_levels=5, head_type='simple', act_type='relu', onnx_export=ONNX_EXPORT,
                 **kwargs):
        assert head_type in ['simple', 'efficient']
        assert act_type in ['relu', 'swish']
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.layers_num = layers_num
        self.num_pyramid_levels = num_pyramid_levels
        logger = kwargs.get('logger', None)
        if logger:
            logger.info(f'==== Build Head Layer ====================')
            logger.info(f'Head Type    : Classification ({head_type} + {act_type})')
            logger.info(f'Features Num : {features_num}')
            logger.info(f'Anchors Num  : {num_anchors}')
            logger.info(f'Layers Num   : {layers_num}')

        _conv_block = SeparableConvBlock if head_type == 'efficient' else nn.Conv2d
        _conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        if head_type == 'efficient':
            _conv_kwargs.update({'norm': False, 'activation': False})
        self.conv_list = nn.ModuleList(
            [_conv_block(in_features_num if i == 0 else features_num, 
                         features_num, **_conv_kwargs) for i in range(layers_num)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(features_num, 
                                           momentum=0.01, eps=1e-3) for i in range(layers_num)]) for j in
             range(num_pyramid_levels)])

        self.header = _conv_block(features_num, num_anchors * num_classes, **_conv_kwargs)

        if act_type == 'swish':
            self.act_fn = MemoryEfficientSwish() if not onnx_export else Swish()
        else:
            self.act_fn = nn.ReLU()
        
        self.header_act = nn.Sigmoid()

        self._initialize_weights(logger=logger)

    def _initialize_weights(self, logger=None):
        if logger:
            logger.info('Initializing weights for Classifier...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.layers_num), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.act_fn(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        # feats = feats.sigmoid()
        feats = self.header_act(feats)

        return feats



class RegressorV1(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_features_num, num_anchors=9, 
                 feature_size=256, num_layers=3, num_pyramid_levels=5, onnx_export=ONNX_EXPORT,
                 **kwargs):
        super(RegressorV1, self).__init__()
        self.num_layers = num_layers
        self.num_pyramid_levels = num_pyramid_levels

        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_features_num, in_features_num, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_features_num, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(num_pyramid_levels)])
        self.header = SeparableConvBlock(in_features_num, num_anchors * 4, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], -1, 4)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)

        return feats


class ClassifierV1(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, in_features_num, num_anchors=9, num_classes=80, 
                 feature_size=256, num_layers=3, num_pyramid_levels=5, onnx_export=ONNX_EXPORT,
                 **kwargs):
        super(ClassifierV1, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_pyramid_levels = num_pyramid_levels
        self.conv_list = nn.ModuleList(
            [SeparableConvBlock(in_features_num, in_features_num, norm=False, activation=False) for i in range(num_layers)])
        self.bn_list = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_features_num, momentum=0.01, eps=1e-3) for i in range(num_layers)]) for j in
             range(num_pyramid_levels)])
        self.header = SeparableConvBlock(in_features_num, num_anchors * num_classes, norm=False, activation=False)
        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.contiguous().view(feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors,
                                          self.num_classes)
            feat = feat.contiguous().view(feat.shape[0], -1, self.num_classes)

            feats.append(feat)

        feats = torch.cat(feats, dim=1)
        feats = feats.sigmoid()

        return feats


'''
Source from: yhenon / pytorch-retinanet
    https://github.com/yhenon/pytorch-retinanet

'''
class RegressorV2(nn.Module):
    def __init__(self, in_features_num, num_anchors=9, feature_size=256):
        super(RegressorV2, self).__init__()
        self.convert_onnx = False
        self.fixed_size = None

        self.conv1 = nn.Conv2d(in_features_num, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

        self._init_weights()


    def _init_weights(self):
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)


    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))
        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        if self.convert_onnx:
            print('Convert ONNX Mode at RegressoinModel')
            return out.contiguous().view(1, -1, 4)
        else:  
            # print('RegressionModel::out.shape[0] = {}'.format(str(out.shape[0])))
            return out.contiguous().view(out.shape[0], -1, 4)


class ClassifierV2(nn.Module):
    def __init__(self, in_features_num, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassifierV2, self).__init__()
        self.convert_onnx = False
        self.fixed_size = None

        self.prior = prior
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_features_num, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        prior = self.prior
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        if self.convert_onnx:
            assert self.fixed_size is not None, 'ClassificationModel::fixed_size Must to be not None.'
            print('Convert ONNX Mode at ClassificationModel')
            print('ClassificationModel::fixed_size = {}'.format(self.fixed_size))
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))
        out = self.output_act(self.output(out))

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