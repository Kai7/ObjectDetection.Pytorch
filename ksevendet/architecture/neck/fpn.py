import torch.nn as nn
import torch.nn.functional as F
import math

# from mmcv.cnn import xavier_init
# from mmdet.core import auto_fp16
# from mmdet.ops import ConvModule

__all__ = ['FPN', 'PANetFPN']

class FPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256, 
                 in_pyramid_levels=[3, 4, 5], out_pyramid_levels=[3, 4, 5, 6, 7]):
        super(FPN, self).__init__()

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
        # for x_feature in inputs:
        #     print(x_feature.shape)
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

        fpn_output_blobs = [P3_x, P4_x, P5_x, P6_x, P7_x]
        # print('old fpn')
        # for x_feature in fpn_output_blobs:
        #     print(x_feature.shape)
        # exit(0)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]

# ---------------------------------------------------------------------------- #

def get_group_gn(dim, dim_per_gp=-1, num_groups=32):
    """
    get number of groups used by GroupNorm, based on number of channels
    """
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0
        group_gn = num_groups
    return group_gn

# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #
class PANetFPN(nn.Module):
    """Add FPN connections based on the model described in the FPN paper.

    fpn_output_blobs is in reversed order: e.g [fpn5, fpn4, fpn3, fpn2]
    similarly for fpn_level_info.dims: e.g [2048, 1024, 512, 256]
    similarly for spatial_scale: e.g [1/32, 1/16, 1/8, 1/4]
    """
    def __init__(self, backbone_features_num, in_pyramid_levels=[3,4,5], out_pyramid_levels=[3,4,5,6,7], 
                 features_num=256, panet_buttomup=False, 
                 group_norm=False, gn_eps=1e-5, dim_per_gp=-1, num_groups=32, **kwargs):
    # def __init__(self, fpn_level_info, P2only=False, panet_buttomup=False):
        super().__init__()
        self.panet_buttomup = panet_buttomup
        self.features_num = features_num
        self.in_pyramid_levels = in_pyramid_levels      # [3, 4, 5]
        self.out_pyramid_levels = out_pyramid_levels    # [3, 4, 5, 6, 7]
        self.backbone_features_num = backbone_features_num[::-1]

        logger = kwargs.get('logger', None)
        if logger:
            logger.info(f'==== Build Neck Layer ====================')
            logger.info('Neck : {}'.format('FPN' if not panet_buttomup else 'PANet-FPN'))
            logger.info(f'FPN Features : {features_num}')

        # self.dim_out = fpn_dim = cfg.FPN.DIM
        # min_level, max_level = get_min_max_levels()

        # self.num_backbone_stages = len(fpn_level_info.blobs) - (min_level - LOWEST_BACKBONE_LVL)
        # self.backbone_features_num = backbone_features_num
        # fpn_dim_lateral = fpn_level_info.dims

        #
        # Step 1: recursively build down starting from the coarsest backbone level
        #
        # For the coarest backbone level: 1x1 conv only seeds recursion
        if group_norm:
            self.conv_top = nn.Sequential(
                nn.Conv2d(self.backbone_features_num[0], self.features_num, 1, 1, 0, bias=False),
                nn.GroupNorm(get_group_gn(self.features_num), self.features_num,
                             eps=gn_eps, dim_per_gp=dim_per_gp, num_groups=num_groups)
            )
        else:
            self.conv_top = nn.Conv2d(self.backbone_features_num[0], self.features_num, 1, 1, 0)

        self.topdown_lateral_modules = nn.ModuleList()
        self.posthoc_modules = nn.ModuleList()

        # For other levels add top-down and lateral connections
        for i in range(1, len(self.in_pyramid_levels)):
            self.topdown_lateral_modules.append(
                topdown_lateral_module(self.features_num, self.backbone_features_num[i])
            )

        # Post-hoc scale-specific 3x3 convs
        for i in range(len(self.in_pyramid_levels)):
            if group_norm:
                self.posthoc_modules.append(nn.Sequential(
                    nn.Conv2d(self.features_num, self.features_num, 3, 1, 1, bias=False),
                    nn.GroupNorm(get_group_gn(self.features_num), self.features_num,
                                 eps=gn_eps, dim_per_gp=dim_per_gp, num_groups=num_groups)
                ))
            else:
                self.posthoc_modules.append(
                    nn.Conv2d(self.features_num, self.features_num, 3, 1, 1)
                )

        # add for panet buttom-up path
        if self.panet_buttomup:
            self.panet_buttomup_conv1_modules = nn.ModuleList()
            self.panet_buttomup_conv2_modules = nn.ModuleList()
            for i in range(len(self.in_pyramid_levels) - 1):
                if group_norm:
                    self.panet_buttomup_conv1_modules.append(nn.Sequential(
                        nn.Conv2d(self.features_num, self.features_num, 3, 2, 1, bias=True),
                        nn.GroupNorm(get_group_gn(self.features_num), self.features_num,
                                     eps=gn_eps, dim_per_gp=dim_per_gp, num_groups=num_groups),
                        nn.ReLU(inplace=True)
                    ))
                    self.panet_buttomup_conv2_modules.append(nn.Sequential(
                        nn.Conv2d(self.features_num, self.features_num, 3, 1, 1, bias=True),
                        nn.GroupNorm(get_group_gn(self.features_num), self.features_num,
                                     eps=gn_eps, dim_per_gp=dim_per_gp, num_groups=num_groups),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    self.panet_buttomup_conv1_modules.append(
                        nn.Conv2d(self.features_num, self.features_num, 3, 2, 1)
                    )
                    self.panet_buttomup_conv2_modules.append(
                        nn.Conv2d(self.features_num, self.features_num, 3, 1, 1)
                    )


        # Coarser FPN levels introduced for RetinaNet
        # if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_BACKBONE_LVL:
        self.extra_levels = self.out_pyramid_levels[-1] > self.in_pyramid_levels[-1]
        if self.extra_levels:
            self.extra_pyramid_modules = nn.ModuleList()
            # dim_in = fpn_level_info.dims[0]
            dim_in = self.backbone_features_num[0] 
            # for i in range(HIGHEST_BACKBONE_LVL + 1, max_level + 1):
            for i in range(self.in_pyramid_levels[-1], self.out_pyramid_levels[-1]):
                self.extra_pyramid_modules.append(
                    nn.Conv2d(dim_in, self.features_num, 3, 2, 1)
                )
                dim_in = self.features_num

        # self._init_weights()
        self._initialize_weights(logger=logger)

    def _initialize_weights(self, logger=None):
        if logger:
            logger.info('Initializing weights for PANetFPN ...')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _init_weights(self):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # mynn.init.XavierFill(m.weight)
                #mynn.init.MSRAFill(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for child_m in self.children():
            if (not isinstance(child_m, nn.ModuleList) or
                not isinstance(child_m[0], topdown_lateral_module)):
                # topdown_lateral_module has its own init method
                child_m.apply(init_func)


    def forward(self, x_backbone_features):
        # for x_feature in x_backbone_features:
        #     print(x_feature.shape)

        # conv_body_blobs = [self.conv_body.res1(x)]
        # for i in range(1, self.conv_body.convX):
        #     conv_body_blobs.append(
        #         getattr(self.conv_body, 'res%d' % (i+1))(conv_body_blobs[-1])
        #     )
        assert len(x_backbone_features) == len(self.in_pyramid_levels), "The number of backbone features don't match FPN pyramid levels"

        fpn_inner_blobs = [self.conv_top(x_backbone_features[-1])]
        for i in range(len(self.in_pyramid_levels) - 1):
            fpn_inner_blobs.append(
                self.topdown_lateral_modules[i](fpn_inner_blobs[-1], x_backbone_features[-(i+2)])
            )
        fpn_output_blobs = []
        if self.panet_buttomup:
            fpn_middle_blobs = []
        for i in range(len(self.in_pyramid_levels)):
            if not self.panet_buttomup:
                fpn_output_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
            else:
                fpn_middle_blobs.append(
                    self.posthoc_modules[i](fpn_inner_blobs[i])
                )
        if self.panet_buttomup:
            fpn_output_blobs.append(fpn_middle_blobs[-1])
            # print(fpn_middle_blobs[-1].shape)
            for i in range(1, len(self.in_pyramid_levels)):
                fpn_tmp = self.panet_buttomup_conv1_modules[i - 1](fpn_output_blobs[0])
                # print(fpn_tmp.shape)
                # print(fpn_middle_blobs[len(self.in_pyramid_levels) - 1 - i].shape)
                #print(fpn_middle_blobs[self.num_backbone_stages - i].size())
                # fpn_tmp = fpn_tmp + fpn_middle_blobs[len(self.in_pyramid_levels) - i]
                fpn_tmp = fpn_tmp + fpn_middle_blobs[len(self.in_pyramid_levels) - 1 - i]
                fpn_tmp = self.panet_buttomup_conv2_modules[i - 1](fpn_tmp)
                fpn_output_blobs.insert(0, fpn_tmp)        

        # if hasattr(self, 'extra_pyramid_modules'):
        if self.extra_levels:
            blob_in = x_backbone_features[-1]
            fpn_output_blobs.insert(0, self.extra_pyramid_modules[0](blob_in))
            for module in self.extra_pyramid_modules[1:]:
                fpn_output_blobs.insert(0, module(F.relu(fpn_output_blobs[0], inplace=True)))
                # fpn_output_blobs.insert(0, module(F.relu(fpn_output_blobs[0], inplace=False)))
        
        # print('new fpn')
        # for x_feature in fpn_output_blobs:
        #     print(x_feature.shape)
        # exit(0)
        return fpn_output_blobs[::-1]


class topdown_lateral_module(nn.Module):
    """Add a top-down lateral module."""
    def __init__(self, dim_in_top, dim_in_lateral, group_norm=False, zero_init_lateral=False, gn_eps=1e-5):
        super().__init__()
        self.dim_in_top = dim_in_top
        self.dim_in_lateral = dim_in_lateral
        self.dim_out = dim_in_top
        self.group_norm = group_norm
        if group_norm:
            self.conv_lateral = nn.Sequential(
                nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(self.dim_out), self.dim_out,
                             eps=gn_eps)
            )
        else:
            self.conv_lateral = nn.Conv2d(dim_in_lateral, self.dim_out, 1, 1, 0)

        self._init_weights(zero_init_lateral=zero_init_lateral)

    def _init_weights(self, zero_init_lateral=False):
        if self.group_norm:
            conv = self.conv_lateral[0]
        else:
            conv = self.conv_lateral

        if zero_init_lateral:
            nn.init.constant_(conv.weight, 0)
        else:
            nn.init.xavier_normal_(conv.weight)
            # mynn.init.XavierFill(conv.weight)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def forward(self, top_blob, lateral_blob):
        # Lateral 1x1 conv
        lat = self.conv_lateral(lateral_blob)
        # Top-down 2x upsampling
        # td = F.upsample(top_blob, size=lat.size()[2:], mode='bilinear')
        # td = F.upsample(top_blob, scale_factor=2, mode='nearest')
        td = F.interpolate(top_blob, scale_factor=2, mode='nearest')
        # Sum lateral and top-down
        return lat + td


# class FPN(nn.Module):
#     """
#     Feature Pyramid Network.

#     This is an implementation of - Feature Pyramid Networks for Object
#     Detection (https://arxiv.org/abs/1612.03144)

#     Args:
#         in_channels (List[int]):
#             number of input channels per scale

#         out_channels (int):
#             number of output channels (used at each scale)

#         num_outs (int):
#             number of output scales

#         start_level (int):
#             index of the first input scale to use as an output scale

#         end_level (int, default=-1):
#             index of the last input scale to use as an output scale

#     Example:
#         >>> import torch
#         >>> in_channels = [2, 3, 5, 7]
#         >>> scales = [340, 170, 84, 43]
#         >>> inputs = [torch.rand(1, c, s, s)
#         ...           for c, s in zip(in_channels, scales)]
#         >>> self = FPN(in_channels, 11, len(in_channels)).eval()
#         >>> outputs = self.forward(inputs)
#         >>> for i in range(len(outputs)):
#         ...     print('outputs[{}].shape = {!r}'.format(i, outputs[i].shape))
#         outputs[0].shape = torch.Size([1, 11, 340, 340])
#         outputs[1].shape = torch.Size([1, 11, 170, 170])
#         outputs[2].shape = torch.Size([1, 11, 84, 84])
#         outputs[3].shape = torch.Size([1, 11, 43, 43])
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  start_level=0,
#                  end_level=-1,
#                  add_extra_convs=False,
#                  extra_convs_on_inputs=True,
#                  relu_before_extra_convs=False,
#                  no_norm_on_lateral=False,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  act_cfg=None):
#         super(FPN, self).__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#         self.relu_before_extra_convs = relu_before_extra_convs
#         self.no_norm_on_lateral = no_norm_on_lateral
#         self.fp16_enabled = False

#         if end_level == -1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level < inputs, no extra level is allowed
#             self.backbone_end_level = end_level
#             assert end_level <= len(in_channels)
#             assert num_outs == end_level - start_level
#         self.start_level = start_level
#         self.end_level = end_level
#         self.add_extra_convs = add_extra_convs
#         self.extra_convs_on_inputs = extra_convs_on_inputs

#         self.lateral_convs = nn.ModuleList()
#         self.fpn_convs = nn.ModuleList()

#         for i in range(self.start_level, self.backbone_end_level):
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                 act_cfg=act_cfg,
#                 inplace=False)
#             fpn_conv = ConvModule(
#                 out_channels,
#                 out_channels,
#                 3,
#                 padding=1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 act_cfg=act_cfg,
#                 inplace=False)

#             self.lateral_convs.append(l_conv)
#             self.fpn_convs.append(fpn_conv)

#         # add extra conv layers (e.g., RetinaNet)
#         extra_levels = num_outs - self.backbone_end_level + self.start_level
#         if add_extra_convs and extra_levels >= 1:
#             for i in range(extra_levels):
#                 if i == 0 and self.extra_convs_on_inputs:
#                     in_channels = self.in_channels[self.backbone_end_level - 1]
#                 else:
#                     in_channels = out_channels
#                 extra_fpn_conv = ConvModule(
#                     in_channels,
#                     out_channels,
#                     3,
#                     stride=2,
#                     padding=1,
#                     conv_cfg=conv_cfg,
#                     norm_cfg=norm_cfg,
#                     act_cfg=act_cfg,
#                     inplace=False)
#                 self.fpn_convs.append(extra_fpn_conv)

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution='uniform')

#     @auto_fp16()
#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels)

#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]

#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             prev_shape = laterals[i - 1].shape[2:]
#             laterals[i - 1] += F.interpolate(
#                 laterals[i], size=prev_shape, mode='nearest')

#         # build outputs
#         # part 1: from original levels
#         outs = [
#             self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#         ]
#         # part 2: add extra levels
#         if self.num_outs > len(outs):
#             # use max pool to get more levels on top of outputs
#             # (e.g., Faster R-CNN, Mask R-CNN)
#             if not self.add_extra_convs:
#                 for i in range(self.num_outs - used_backbone_levels):
#                     outs.append(F.max_pool2d(outs[-1], 1, stride=2))
#             # add conv layers on top of original feature maps (RetinaNet)
#             else:
#                 if self.extra_convs_on_inputs:
#                     orig = inputs[self.backbone_end_level - 1]
#                     outs.append(self.fpn_convs[used_backbone_levels](orig))
#                 else:
#                     outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
#                 for i in range(used_backbone_levels + 1, self.num_outs):
#                     if self.relu_before_extra_convs:
#                         outs.append(self.fpn_convs[i](F.relu(outs[-1])))
#                     else:
#                         outs.append(self.fpn_convs[i](outs[-1]))
#         return tuple(outs)
