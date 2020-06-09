import torch.nn as nn
import torch
#from torchvision.ops.boxes import nms as nms_torch
from torchvision.ops import nms
import math

from architectures.efficientnet import EfficientNet as EffNet
from architectures.utils_efficientnet import MemoryEfficientSwish, Swish
from architectures.utils_extra_efficientnet import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

import pdb

class BiFPN(nn.Module):
    """
    modified by Zylo117
    """
    # TODO: Support specific in_pyramid_levels and out_pyramid_levels 

    # def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
    def __init__(self, backbone_features_num, in_pyramid_levels=[3,4,5], out_pyramid_levels=[3,4,5,6,7], features_num=64, 
                 first_time=False, epsilon=1e-4, onnx_export=True, attention=False, act_func='relu', **kwargs):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        assert in_pyramid_levels  == [3,4,5]
        assert out_pyramid_levels == [3,4,5,6,7]
        assert act_func in ['relu', 'swish'], f'Unknown act_func: {act_func}'
        super(BiFPN, self).__init__()
        self.convert_onnx = False
        self.pyramid_sizes = None

        self.backbone_features_num = backbone_features_num
        self.in_pyramid_levels  = in_pyramid_levels
        self.out_pyramid_levels = out_pyramid_levels
        self.features_num = features_num
        self.epsilon = epsilon
        self.attention = attention

        logger = kwargs.get('logger', None)
        if logger:
            logger.info('Build Neck: BiFPN')
            logger.info(f'Features Num: {features_num}')
            logger.info(f'First: {first_time}')
            logger.info(f'Attention: {attention}')

        # Conv layers
        self.conv3_up = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv6_up = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(features_num, activation=True, act_func=act_func, onnx_export=onnx_export)
        # print(self.conv6_up)
        # print(self.conv4_down)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.p4_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p5_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p6_downsample = nn.MaxPool2d(3, stride=2, padding=1)
        self.p7_downsample = nn.MaxPool2d(3, stride=2, padding=1)

        # self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        if act_func == 'swish':
            self.act_func = MemoryEfficientSwish() if not onnx_export else Swish()
        elif act_func == 'relu':
            self.act_func = nn.ReLU()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                # Conv2dStaticSamePadding(conv_channels[2], features_num, 1),
                # nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
                nn.Conv2d(backbone_features_num[2], features_num, 1),
                nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(backbone_features_num[1], features_num, 1),
                nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(backbone_features_num[0], features_num, 1),
                nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(backbone_features_num[2], features_num, 1),
                nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(3, stride=2, padding=1)
            )

            self.p4_down_channel_2 = nn.Sequential(
                nn.Conv2d(backbone_features_num[1], features_num, 1),
                nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                nn.Conv2d(backbone_features_num[2], features_num, 1),
                nn.BatchNorm2d(features_num, momentum=0.01, eps=1e-3),
            )

        # Weight
        if self.attention:
            self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p6_w1_relu = nn.ReLU()
            self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p5_w1_relu = nn.ReLU()
            self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p4_w1_relu = nn.ReLU()
            self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p3_w1_relu = nn.ReLU()

            self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p4_w2_relu = nn.ReLU()
            self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p5_w2_relu = nn.ReLU()
            self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
            self.p6_w2_relu = nn.ReLU()
            self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.p7_w2_relu = nn.ReLU()

        self._initialize_weights(logger=logger)

    def _initialize_weights(self, logger=None):
        if logger:
            logger.info('Initializing weights for BiFPN ...')

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
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation
        # print('BiFPN forward...')
        if self.attention:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
        else:
            p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.act_func(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_0 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.act_func(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_0 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.act_func(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.act_func(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.act_func(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.act_func(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.act_func(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.act_func(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.convert_onnx:
            assert self.pyramid_sizes is not None, 'NECK[BiFPN]::pyramid_sizes must be not None'
            print('Convert ONNX Mode at NECK[BiFPN]')
            for i in range(len(self.pyramid_sizes)):
                print('[{}] Pyramid Feature Grid Size = [{:>4},{:>4}]'.format(i, *self.pyramid_sizes[i]))

        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2
        if self.convert_onnx:
            self.p6_upsample.scale_factor = None
            self.p5_upsample.scale_factor = None
            self.p4_upsample.scale_factor = None
            self.p3_upsample.scale_factor = None
            self.p6_upsample.size = list(self.pyramid_sizes[3])
            self.p5_upsample.size = list(self.pyramid_sizes[2])
            self.p4_upsample.size = list(self.pyramid_sizes[1])
            self.p3_upsample.size = list(self.pyramid_sizes[0])

        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.act_func(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_0 to P5_1 respectively
        p5_up = self.conv5_up(self.act_func(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_0 to P4_1 respectively
        p4_up = self.conv4_up(self.act_func(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.act_func(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.act_func(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.act_func(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.act_func(p6_in + p6_up + self.p6_downsample(p5_out)))

        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.act_func(p7_in + self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out



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

    # def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=True, act_func='relu', onnx_export=True):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        # self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
        #                                               kernel_size=3, stride=1, groups=in_channels, bias=False)
        # self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            if act_func == 'swish':
                self.act_func = MemoryEfficientSwish() if not onnx_export else Swish()
            elif act_func == 'relu':
                self.act_func = nn.ReLU()
            else:
                assert 0, f'Unknown act_func: {act_func}'

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.act_func(x)

        return x


# class Conv2dStaticSamePadding(nn.Module):
#     """
#     created by Zylo117
#     The real keras/tensorflow conv2d with same padding
#     """

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
#                               bias=bias, groups=groups)
#         self.stride = self.conv.stride
#         self.kernel_size = self.conv.kernel_size
#         self.dilation = self.conv.dilation

#         if isinstance(self.stride, int):
#             self.stride = [self.stride] * 2
#         elif len(self.stride) == 1:
#             self.stride = [self.stride[0]] * 2

#         if isinstance(self.kernel_size, int):
#             self.kernel_size = [self.kernel_size] * 2
#         elif len(self.kernel_size) == 1:
#             self.kernel_size = [self.kernel_size[0]] * 2

#     def forward(self, x):
#         h, w = x.shape[-2:]

#         h_step = math.ceil(w / self.stride[1])
#         v_step = math.ceil(h / self.stride[0])
#         h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
#         v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

#         extra_h = h_cover_len - w
#         extra_v = v_cover_len - h

#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top

#         x = F.pad(x, [left, right, top, bottom])

#         x = self.conv(x)
#         return x


# class MaxPool2dStaticSamePadding(nn.Module):
#     """
#     created by Zylo117
#     The real keras/tensorflow MaxPool2d with same padding
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.pool = nn.MaxPool2d(*args, **kwargs)
#         self.stride = self.pool.stride
#         self.kernel_size = self.pool.kernel_size

#         if isinstance(self.stride, int):
#             self.stride = [self.stride] * 2
#         elif len(self.stride) == 1:
#             self.stride = [self.stride[0]] * 2

#         if isinstance(self.kernel_size, int):
#             self.kernel_size = [self.kernel_size] * 2
#         elif len(self.kernel_size) == 1:
#             self.kernel_size = [self.kernel_size[0]] * 2

#     def forward(self, x):
#         h, w = x.shape[-2:]

#         h_step = math.ceil(w / self.stride[1])
#         v_step = math.ceil(h / self.stride[0])
#         h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
#         v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

#         extra_h = h_cover_len - w
#         extra_v = v_cover_len - h

#         left = extra_h // 2
#         right = extra_h - left
#         top = extra_v // 2
#         bottom = extra_v - top

#         x = F.pad(x, [left, right, top, bottom])

#         x = self.pool(x)
#         return x