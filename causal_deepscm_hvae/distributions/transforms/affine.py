from pyro.distributions.conditional import ConditionalTransformModule
from pyro.distributions.torch_transform import TransformModule
from pyro.distributions import transforms as pyro_transforms
from torch.distributions import transforms

import torch


class LearnedAffineTransform(TransformModule, transforms.AffineTransform):
    def __init__(self, loc=None, scale=None, **kwargs):

        super().__init__(loc=loc, scale=scale, **kwargs)

        if loc is None:
            self.loc = torch.nn.Parameter(torch.zeros([1, ]))
        if scale is None:
            self.scale = torch.nn.Parameter(torch.ones([1, ]))

    def _broadcast(self, val):
        dim_extension = tuple(1 for _ in range(val.dim() - 1))
        loc = self.loc.view(-1, *dim_extension)
        scale = self.scale.view(-1, *dim_extension)

        return loc, scale

    def _call(self, x):
        loc, scale = self._broadcast(x)

        return loc + scale * x

    def _inverse(self, y):
        loc, scale = self._broadcast(y)
        return (y - loc) / scale


class ConditionalAffineTransform(ConditionalTransformModule):
    def __init__(self, context_nn, event_dim=0,
                 min_scale=1e-3, max_scale=1e3, max_shift=5e1, **kwargs):
        super().__init__(**kwargs)
        self.event_dim = event_dim
        self.context_nn = context_nn
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_shift = float(max_shift)

    def condition(self, context):
        loc, log_scale = self.context_nn(context)          # DenseNN 返回两个张量
        # 1) 稳定的正数参数化 + 下界
        scale = torch.nn.functional.softplus(log_scale) + self.min_scale
        # 2) 上界裁剪，避免极端值
        scale = torch.clamp(scale, max=self.max_scale)
        # 3) shift 也限制一下幅度（可选）
        loc = torch.clamp(loc, -self.max_shift, self.max_shift)

        return transforms.AffineTransform(loc, scale, event_dim=self.event_dim)



class LowerCholeskyAffine(pyro_transforms.LowerCholeskyAffine):
    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log Jacobian, i.e.
        log(abs(dy/dx)).
        """
        return torch.ones(x.size()[:-1], dtype=x.dtype, layout=x.layout, device=x.device) * \
            self.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1).sum(-1)
