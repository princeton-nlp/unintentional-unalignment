from .callback import *


class RequiresGradChange(Callback):

    def __init__(self, params, epoch, requires_grad=True):
        """
        :param params: sequence of parameters.
        :param epoch: epoch number to change requires grad value of parameters on start of.
        """
        self.params = params
        self.epoch = epoch
        self.requires_grad = requires_grad

    def on_epoch_start(self, trainer):
        if trainer.epoch == self.epoch:
            for param in self.params:
                param.requires_grad = self.requires_grad
