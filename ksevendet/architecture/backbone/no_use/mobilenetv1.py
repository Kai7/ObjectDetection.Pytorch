"""
Creates a MobileNetV1 Model:
modified from  https://github.com/ruotianluo/pytorch-mobilenet-from-tf 
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple, OrderedDict, Iterable
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .registry import register_model
from .helpers import load_pretrained
from .feature_hooks import FeatureHooks
from collections import OrderedDict
from .adaptive_avgmax_pool import SelectAdaptivePool2d


__all__ = ['MobileNetV1']

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv0', 'classifier': 'classifier',
        **kwargs
    }

default_cfgs = {
    'mobilenetv1_100': _cfg(url=''),
    'mobilenetv1_75': _cfg(
        url=''),
    'mobilenetv1_50': _cfg(
        url=''),
    'mobilenetv1_25': _cfg(
        url=''),


    # mobilenet v1 tensorflow version
    # Pretrained pytorch model converted from tensorflow, copy from https://github.com/ruotianluo/pytorch-mobilenet-from-tf
    # TODO: use internal ftp instead of local file url here!
    'mobilenetv1_100_tf': _cfg(
        url='file://localhost/dataset/work/algorithms/model_zoo/pretrained/mobilenet_v1_tf_1.0_224.pth',
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD),
    'mobilenetv1_75_tf': _cfg(
        url='file://localhost/dataset/work/algorithms/model_zoo/pretrained/mobilenet_v1_tf_0.75_224.pth',
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD),
    'mobilenetv1_50_tf': _cfg(
        url='file://localhost/dataset/work/algorithms/model_zoo/pretrained/mobilenet_v1_tf_0.5_224.pth',
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD),
    'mobilenetv1_25_tf': _cfg(
        url='file://localhost/dataset/work/algorithms/model_zoo/pretrained/mobilenet_v1_tf_0.25_224.pth',
        mean=IMAGENET_INCEPTION_MEAN,
        std=IMAGENET_INCEPTION_STD),
}

class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')
        kwargs['padding'] = 0
        if not isinstance(self.stride, Iterable):
            self.stride = (self.stride, self.stride)
        if not isinstance(self.dilation, Iterable):
            self.dilation = (self.dilation, self.dilation)

    def forward(self, input):
        # from https://github.com/pytorch/pytorch/issues/3867
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding=0,
                            dilation=self.dilation, groups=self.groups)
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows -
                                input_rows)
        # padding_rows = max(0, (out_rows - 1) * self.stride[0] +
        #                         (filter_rows - 1) * self.dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        # same for padding_cols
        input_cols = input.size(3)
        filter_cols = self.weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols -
                                input_cols)
        # padding_cols = max(0, (out_cols - 1) * self.stride[1] +
        #                         (filter_cols - 1) * self.dilation[1] + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])


class _conv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride, Conv2d=nn.Conv2d):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(inp, oup, kernel, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)

class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride, Conv2d=nn.Conv2d):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Sequential(
                Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True)
            ),
            # pw
            nn.Sequential(
                Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


def mobilenet_base(conv_defs, depth=lambda x: x, in_channels=3, Conv2d=nn.Conv2d):
    layers = []
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.kernel, conv_def.stride, Conv2d)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride, Conv2d)]
            in_channels = depth(conv_def.depth)
    return layers, in_channels

class MobileNetV1(nn.Module):
    def __init__(self, depth_multiplier=1.0, min_depth=8, num_classes=1000, in_chans=3, dropout=0.2, global_pool='avg', Conv2d=nn.Conv2d):
        nn.Module.__init__(self)
        BackboneInterface.__init__(self)

        self.dropout = dropout
        conv_defs = [
            Conv(kernel=3, stride=2, depth=32),
            DepthSepConv(stride=1, depth=64),
            DepthSepConv(stride=2, depth=128),
            DepthSepConv(stride=1, depth=128),
            DepthSepConv(stride=2, depth=256),
            DepthSepConv(stride=1, depth=256),
            DepthSepConv(stride=2, depth=512),
            DepthSepConv(stride=1, depth=512),
            DepthSepConv(stride=1, depth=512),
            DepthSepConv(stride=1, depth=512),
            DepthSepConv(stride=1, depth=512),
            DepthSepConv(stride=1, depth=512),
            DepthSepConv(stride=2, depth=1024),
            DepthSepConv(stride=1, depth=1024)
        ]

        depth = lambda d: max(int(d * depth_multiplier), min_depth)
        self.features, self.out_channels = mobilenet_base(conv_defs=conv_defs, depth=depth, in_channels=in_chans, Conv2d=Conv2d)

        self.features = nn.Sequential(*self.features)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Conv2d(self.out_channels, num_classes, 1)
        self.num_classes = num_classes
        for m in self.modules():
            if 'BatchNorm' in m.__class__.__name__:
                m.eps = 0.001
                m.momentum = 0.003

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        del self.classifier
        self.classifier = nn.Conv2d(self.out_channels, num_classes, 1)

    def forward(self, x):
        x = self.forward_features(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.classifier(x)
        x = x.squeeze(3).squeeze(2)
        return x
        
    def forward_features(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return x

def _create_model(variant, width_mult, pretrained, Conv2d=nn.Conv2d, **kwargs):
    model = MobileNetV1(depth_multiplier=width_mult, Conv2d=Conv2d, **kwargs)
    model.default_cfg = default_cfgs[variant]

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
def mobilenetv1_100(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_100', 1.0, pretrained, **kwargs)

@register_model
def mobilenetv1_75(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_75', 0.75, pretrained, **kwargs)

@register_model
def mobilenetv1_50(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_50', 0.5, pretrained, **kwargs)

@register_model
def mobilenetv1_25(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_25', 0.25, pretrained, **kwargs)


@register_model
def mobilenetv1_100_tf(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_100_tf', 1.0, pretrained, Conv2d_tf, **kwargs)

@register_model
def mobilenetv1_75_tf(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_75_tf', 0.75, pretrained, Conv2d_tf, **kwargs)

@register_model
def mobilenetv1_50_tf(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_50_tf', 0.5, pretrained, Conv2d_tf, **kwargs)

@register_model
def mobilenetv1_25_tf(pretrained=False, **kwargs):
    """
    Constructs a MobileNet v1 model
    """
    return _create_model('mobilenetv1_25_tf', 0.25, pretrained, Conv2d_tf, **kwargs)
