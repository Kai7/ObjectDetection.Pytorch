import numpy as np
import torch
import torch.nn as nn

import pdb
# import logging

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None, **kwargs):
        super(Anchors, self).__init__()

        self.pyramid_levels = [3, 4, 5, 6, 7] if pyramid_levels is None else pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels] if strides is None else strides
        if 'sizes_mapper' in kwargs:
            sizes_mapper = eval(kwargs['sizes_mapper'])
            self.sizes   = [sizes_mapper(x) for x in self.pyramid_levels]
        else:
            self.sizes   = [2 ** (x + 2) for x in self.pyramid_levels] if sizes is None else sizes
        self.ratios  = np.array([0.5, 1, 2]) if ratios is None else np.array(ratios)
        if isinstance(scales, str):
            scales = eval(scales)
        self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]) if scales is None else np.array(scales)

        # print('Anchors Sizes =', self.sizes)
        logger = kwargs.get('logger', None)
        if logger:
            kwargs['logger'].info('Anchors Sizes   : [ {} ]'.format(
                                  ', '.join(['{:>4}'.format(int(ss)) for ss in self.sizes])))
            kwargs['logger'].info('Anchors Scales  : [ {} ]'.format(
                                  ', '.join(['{:>4.3f}'.format(ss) for ss in self.scales])))
            kwargs['logger'].info('Anchors Ratios  : [ {} ]'.format(
                                  ', '.join(['{:>4.3f}'.format(rr) for rr in self.ratios])))
            kwargs['logger'].info('Anchors Num     : {}'.format(len(self.ratios) * len(self.scales)))
            for idx in range(len(self.sizes)):
                anchors = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales, get_sample=True)
                anchors_info = list()
                ratio_nums_ = len(self.ratios)
                scale_nums_ = len(self.scales)
                for i_ratio in range(ratio_nums_):
                    anchors_info.append(' '.join(['({:>4},{:>4})'.format(
                                        int(anchors[i_ratio*scale_nums_ + j_scale,2]), int(anchors[i_ratio*scale_nums_ + j_scale,3])) 
                                        for j_scale in range(scale_nums_)]))
                anchors_info = '\n'.join(anchors_info)
                #kwargs['logger'].info('Anchors [{idx}]\n{anchors}'.format(idx=idx, anchors=str(anchors)))
                kwargs['logger'].info('Anchors [{idx}]\n{anchors}'.format(idx=idx, anchors=anchors_info))
                

    def forward(self, image):
        
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        #image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        image_shapes = [(image_shape + x - 1) // x for x in self.strides]
        # print('anchor forward')
        # print(image_shapes)

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def generate_anchors(base_size=16, ratios=None, scales=None, get_sample=False):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    if get_sample:
        return anchors
    
    # pdb.set_trace()
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors

def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors



