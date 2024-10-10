from typing import Union, Dict

import torch.nn as nn
from torch import Tensor


class DataParallelPassthrough(nn.DataParallel):
    """
    DataParallel extension that allows seamless access to the underlying modules attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict: Union[Dict[str, Tensor], Dict[str, Tensor]], strict: bool = True):
        return self.module.load_state_dict(state_dict, strict=strict)
