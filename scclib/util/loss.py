from typing import Optional
from distributed.protocol import torch
from fastai.losses import *
from fastcore.meta import delegates, use_kwargs_dict
from torch import Tensor
from torch.autograd.grad_mode import F
from torch.nn.modules import loss


class smooth_binary_cross_entropy(loss._Loss):
    
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(smooth_binary_cross_entropy, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        c = target.shape[1]
        eps = 0.1
        smoothed_target = torch.where(target == 1, 1 - (eps + (eps / c)), eps / c)
        return F.binary_cross_entropy_with_logits(input,
                                                  smoothed_target,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


@delegates()
class smooth_loss_v2(BaseLoss):
    
    @use_kwargs_dict(keep=True, weight=None, reduction='mean', pos_weight=None)
    def __init__(self, *args, axis=-1, floatify=True, thresh=0.5, **kwargs):
        if kwargs.get('pos_weight', None) is not None and kwargs.get('flatten', None) is True:
            raise ValueError(
                "`flatten` must be False when using `pos_weight` to avoid a RuntimeError due to shape mismatch")
        if kwargs.get('pos_weight', None) is not None: kwargs['flatten'] = False
        super().__init__(smooth_binary_cross_entropy, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.thresh = thresh
    
    def decodes(self, x):
        return x > self.thresh
    
    def activation(self, x):
        return torch.sigmoid(x)
