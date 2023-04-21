import torch
from torch import nn
from torch.nn import functional as F

from .models import register_generator
import pdb

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers

    Taken from https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/anchor_generator.py
    """

    def __init__(self, buffers):
        super().__init__()
        for i, buffer in enumerate(buffers):
            # Use non-persistent buffer so the values are not saved in checkpoint
            self.register_buffer(str(i), buffer, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())

@register_generator('point')
class PointGenerator(nn.Module):
    """
        A generator for temporal "points"

        max_seq_len can be much larger than the actual seq length
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length that the generator will buffer
        fpn_levels,         # number of fpn levels
        scale_factor,       # scale factor between two fpn levels
        regression_range,   # regression range (on feature grids)
        use_offset=False    # if to align the points at grid centers
    ):
        super().__init__()
        # sanity check, # fpn levels and length divisible
        assert len(regression_range) == fpn_levels
        assert max_seq_len % scale_factor**(fpn_levels - 1) == 0

        # save params
        self.max_seq_len = max_seq_len                                      # 2304 * 4
        self.fpn_levels = fpn_levels                                        # 6
        self.scale_factor = scale_factor                                    # 2
        self.regression_range = regression_range                            # [[0, 4], [2, 8], [4, 16], [8, 32], [16, 64], [32, 10000]]
        self.use_offset = use_offset                                        # false
        # generate all points and buffer the list
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        points_list = []
        # initial points
        initial_points = torch.arange(0, self.max_seq_len, 1.0)             # 9216

        
        # loop over all points at each pyramid level
        for l in range(self.fpn_levels):
            stride = self.scale_factor ** l                                            # 0:2^0=1
            reg_range = torch.as_tensor(self.regression_range[l], dtype=torch.float)   # 0:([0., 4.]) 
            fpn_stride = torch.as_tensor(stride, dtype=torch.float)                    # 0:(1.)
            points = initial_points[::stride][:, None]                                 # 0:[9216, 1] 1:[4608, 1]
            # add offset if necessary (not in our current model)
            if self.use_offset:
                points += 0.5 * stride                                                 # 0: [0.5,1.5,...,9215.5]
            # pad the time stamp with additional regression range / stride
            reg_range  = reg_range[None].repeat(points.shape[0], 1)                    # 0:[1, 9216, 2] ---> [9216, 2]
            fpn_stride = fpn_stride[None].repeat(points.shape[0], 1)                   # 0:[1, 9216, 1] ---> [9216, 1]
                                                                                           # value: 0: [1.,1.,...,1.]
            # size: T x 4 (ts, reg_range, stride)
            points_list.append(torch.cat((points, reg_range, fpn_stride), dim=1))      # 0:[9216, 4] 1:[4608, 4] 2:[2304, 4] 3:[1152, 4] 4:[576, 4] 5:[288, 4]
                                                                                       #value0 : [0.0000e+00, 0.0000e+00, 8.0000e+00, 1.0000e+00],
                                                                                       #         [1.0000e+00, 0.0000e+00, 8.0000e+00, 1.0000e+00],
                                                                                       #         [2.0000e+00, 0.0000e+00, 8.0000e+00, 1.0000e+00],
                                                                                       #          ...,
                                                                                       #         [9.2150e+03, 0.0000e+00, 8.0000e+00, 1.0000e+00]
                                                                                       #        
        return BufferList(points_list)                                  

    def forward(self, feats):
        # feats will be a list of torch tensors
        assert len(feats) == self.fpn_levels                                           # feats:0 --- [1, 512, 2304]
        # # pdb.set_trace()
        # for x in feats:
        #     print("feats : ", x.shape)
        pts_list = []
        feat_lens = [feat.shape[-1] for feat in feats]                                 # 2304
        for feat_len, buffer_pts in zip(feat_lens, self.buffer_points):
            # print("feat_len shape: {} buffer_pts shape: {} !".format(feat_len, buffer_pts.shape))
            assert feat_len <= buffer_pts.shape[0], "Reached max buffer length for point generator"
            # pdb.set_trace()
            pts = buffer_pts[:feat_len, :]                                             # [2304, 4]  [1152, 4] [576, 4] [288, 4] [144, 4] [72, 4]
            pts_list.append(pts)
        return pts_list
