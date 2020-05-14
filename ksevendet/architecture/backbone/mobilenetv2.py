"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from ksevendet.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .registry import register_model
from .helpers import load_pretrained
from .feature_hooks import FeatureHooks
from collections import OrderedDict
# from .adaptive_avgmax_pool import SelectAdaptivePool2d
from .layers.adaptive_avgmax_pool import SelectAdaptivePool2d

__all__ = ['MobileNetV2', 'MobileNetV2Features']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv0', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = {
    'mobilenetv2_200': _cfg(url=''),
    'mobilenetv2_140': _cfg(url=''),
    'mobilenetv2_100': _cfg(url='https://github.com/d-li14/mobilenetv2.pytorch/raw/master/pretrained/mobilenetv2_1.0-0c6065bc.pth'),
    'mobilenetv2_75': _cfg(
        url='https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_0.75-dace9791.pth'),
    'mobilenetv2_50': _cfg(
        url='https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_0.5-eaa6f9ad.pth'),
    'mobilenetv2_35': _cfg(
        url='https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_0.35-b2e15951.pth'),
    'mobilenetv2_25': _cfg(
        url='https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_0.25-b61d2159.pth'),
    'mobilenetv2_10': _cfg(
        url='https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2_0.1-7d1d638a.pth'),
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., in_chans=3, out_features=[], drop_rate=0., drop_connect_rate=0., global_pool='avg'):
        super(MobileNetV2, self).__init__()
        self.width_mult = width_mult
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(in_chans, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
                
        self.features = nn.Sequential(*layers)
        self.out_features = out_features

        if len(self.out_features) <= 0:
            # building classifier layers
            output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
            self.conv = conv_1x1_bn(input_channel, output_channel)
            self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
            self.classifier = nn.Linear(output_channel, num_classes)
            self.features_only = False
        else:
            # ignore classifier layer and hook necessary features
            self.features_only = True
            hooks = [dict(name=feature_name, type="forward") for feature_name in self.out_features]
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

        self._initialize_weights()

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        del self.classifier
        del self.global_pool
        self.num_classes = num_classes
        # self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        output_channel = _make_divisible(1280 * self.width_mult, 4 if self.width_mult == 0.1 else 8) if self.width_mult > 1.0 else 1280
        self.classifier = nn.Linear(output_channel, num_classes)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        if self.features_only:
            ret = OrderedDict()
            outputs = self.feature_hooks.get_output(x.device)[::-1]
            for feat_name in self.out_features:
                ret[feat_name] = outputs[self.out_features.index(feat_name)]
            return ret
        else:
            x = self.conv(x)
            x = self.global_pool(x)
            x = x.view(-1, 1280)
            x = self.classifier(x)
            return x

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
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV2Features(nn.Module):
    def __init__(self, width_mult=1., in_chans=3, out_features=[], drop_rate=0., drop_connect_rate=0., global_pool='avg',
                 **kwargs):
        super(MobileNetV2Features, self).__init__()
        logger = kwargs.get('logger', None)
        if logger:
            logger.info('==== Build Backbone ======================')
            logger.info('Backbone   : {}'.format('mobilenetv2'))
            logger.info('Width Mult : {}'.format(width_mult))

        self.width_mult = width_mult
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        # layers = [conv_3x3_bn(in_chans, input_channel, 2)]
        self.stem = conv_3x3_bn(in_chans, input_channel, 2)
        self.feature_blocks = nn.ModuleList()
        # building inverted residual blocks
        block = InvertedResidual
        RETURN_IDXES = [2, 4, 6]
        _feature_map_channels = list()
        for i, (t, c, n, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if i in RETURN_IDXES:
                _feature_map_channels.append(output_channel)
            blocks_ = list()
            for i in range(n):
                # layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                blocks_.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
            self.feature_blocks.append(nn.Sequential(*blocks_))

        # _feature_map_channels = [32, 96, 320]
        self.features_num = dict()
        for i in range(len(_feature_map_channels)):
            self.features_num[f'c{i+3}'] = _feature_map_channels[i]
        self.out_indices = [f'c{idx}' for idx in [3, 4, 5]]

        self._initialize_weights(logger=logger)

    def _initialize_weights(self, logger=None):
        if logger:
            logger.info('Initializing weights for MobileNetV2 ...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def feature_channels(self):
        return [self.features_num[idx] for idx in self.out_indices]

    def forward(self, x):
        # print(f'Input Shape : {str(x.shape)}')
        x_feature = self.stem(x)
        # print(f'x_stem Shape : {str(x.shape)}')
        RETURN_IDXES = [2, 4, 6]

        feature_maps = list()
        for i, _block in enumerate(self.feature_blocks):
            x_feature = _block(x_feature)
            # print(f'Block_{i} Shape : {str(x_feature.shape)}')
            if i in RETURN_IDXES:
                feature_maps.append(x_feature)
        # x_feature = self.final_conv(x_feature)
        # print(f'x_final Shape : {str(x_feature.shape)}')

        return feature_maps


def _create_model(variant, width_mult, pretrained, **kwargs):
    #model.default_cfg = default_cfgs[variant]
    #model = MobileNetV2(width_mult=width_mult, **kwargs)

    default_cfg = default_cfgs[variant]
    if kwargs.pop('features_only', False):
        model = MobileNetV2Features(width_mult=width_mult, **kwargs)
    else:
        model = MobileNetV2(width_mult=width_mult, **kwargs)
    model.default_cfg = default_cfg

    if pretrained:
        model_kwargs = dict(
            **kwargs
        )
        load_pretrained(
            model,
            model.default_cfg,
            num_classes=model_kwargs.get('num_classes', 0),
            strict=not model.features_only)
    return model

@register_model
def mobilenetv2_200(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_200', 1.4, pretrained, **kwargs)

@register_model
def mobilenetv2_140(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_140', 1.4, pretrained, **kwargs)

@register_model
def mobilenetv2_100(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    print('----')
    return _create_model('mobilenetv2_100', 1.0, pretrained, **kwargs)

@register_model
def mobilenetv2_75(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_75', 0.75, pretrained, **kwargs)

@register_model
def mobilenetv2_50(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_50', 0.5, pretrained, **kwargs)

@register_model
def mobilenetv2_35(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_35', 0.35, pretrained, **kwargs)

@register_model
def mobilenetv2_25(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_25', 0.25, pretrained, **kwargs)


@register_model
def mobilenetv2_10(pretrained=False, **kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return _create_model('mobilenetv2_10', 0.1, pretrained, **kwargs)
