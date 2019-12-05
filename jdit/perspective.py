import os
from typing import Union, Dict, Tuple
from jdit.model import Model
from jdit.trainer.super import Watcher
from torch.nn import Module
import torch
from torch import Tensor


class FeatureVisualization(object):
    """ Visualize the activations of model

    This class can visualize the activations of each layer in model.
    :param model:model network. It can be ``jdit.Model`` or ``torch.nn.Module``
    :param log:The log directory and ``--logdir=`` of tensorboard will be run in this dir.
    """

    def __init__(self, model: Union[Model, Module], logdir: str):
        model.eval()
        self.model = model
        self.activations: Dict[str, Tensor] = {}
        self.watcher = Watcher(logdir=logdir)

    def _hook(self, name: str):
        def hook(model, input, output):
            self.activations[name] = output.detach()

        return hook

    @property
    def layers_names(self):
        """Get model's layers names.

        :return:
        """
        layer_names = []
        for layer in self.model.named_modules():
            layer_names.append(layer[0])
        return layer_names

    def _register_forward_hook(self, block_name: str):
        """Set a forward hook in model.

        :param block_name: block name of model
        :return:
        """
        if not hasattr(self.model, block_name):
            raise AttributeError(
                    "model doesn't have `%s` layer. These layers available: %s" % (
                        block_name, str.join("\n", self.layers_names)))

        getattr(self.model, block_name).register_forward_hook(self._hook(block_name))

    def trace_activation(self, input_tensor: Union[Tensor, Tuple[Tensor]], block_names: list,
                         use_gpu=False) -> \
            Dict[str, Tensor]:
        """Trace the activations of model during a forward propagation.

        :param input_tensor:Input of model. Tensor or tuple of Tensors
        :param block_names:A list of layer's key names which you want to visualize.
        :param use_gpu:If use GPU. Default: False
        :return:The activations dic with block_names as the keys.
        """
        if len(input_tensor.shape) == 4 and input_tensor.shape[0] != 1:
            raise ValueError(
                    "You can only pass one sample to do feature visualization, but %d was given" %
                    input_tensor.shape[0])

        for name in block_names:
            self._register_forward_hook(name)
        with torch.no_grad():
            if use_gpu:
                self.model = self.model.cuda()
                if isinstance(input_tensor, Tensor):
                    input_tensor = input_tensor.cuda()
                    self.model(input_tensor)
                elif isinstance(input_tensor, (tuple, list)):
                    input_tensor = [item.cuda() for item in input_tensor]
                    self.model(*input_tensor)
                else:
                    raise TypeError(
                            "`input_tensor` should be Tensor or tuple of Tensors, but %s was given" % type(
                                    input_tensor))
            else:
                if isinstance(input_tensor, Tensor):
                    self.model(input_tensor)
                elif isinstance(input_tensor, (tuple, list)):
                    self.model(*input_tensor)
                else:
                    raise TypeError(
                            "`input_tensor` should be Tensor or tuple of Tensors, but %s was given" % type(
                                    input_tensor))

        return self.activations

    def show_feature_trace(self, input_tensor: Union[Tensor, Tuple[Tensor]], block_names: list, global_step,
                           grid_size=(4, 4), show_struct=True, use_gpu=False, shuffle=False):
        """Trace the activations of model during a forward propagation.

        :param input_tensor: Input of model. Tensor or tuple of Tensors
        :param block_names: A list of layer's key names which you want to visualize.
        :param global_step:step flag.
        :param grid_size:Amount and size of actives images which you want to show in tensorboard.
        :param show_struct:If show structure of model. Default: True
        :param use_gpu:If use GPU. Default: False
        :param shuffle:If shuffle the activations channels. Default: False
        :return:

        Example:

            .. code-block:: python

                from torch.nn import Conv2d
                class CL(Module):
                    def __init__(self):
                        super(CL, self).__init__()
                        self.layer_1 = Conv2d(1, 2, 3, 1, 1)
                        self.layer_2 = Conv2d(2, 4, 3, 1, 1)
                        self.layer_3 = Conv2d(4, 8, 3, 1, 1)
                        self.layer_4 = Conv2d(8, 16, 3, 1, 1)
                    def forward(self, input):
                        out = self.layer_1(input)
                        out = self.layer_2(out)
                        out = self.layer_3(out)
                        out = self.layer_4(out)
                        return out

                model = CL()
                fv = FeatureVisualization(model, "incep_test")
                input = torch.randn(1, 1, 10, 10)
                fv.show_feature_trace(input, ["layer_1", "layer_2", "layer_3"], 1)
                fv.watcher.close()

        """
        self.trace_activation(input_tensor, block_names)
        if show_struct:
            self.watcher.graph(self.model, self.model.__class__.__name__, use_gpu, input_tensor.size())

        for layer_name, tensor in self.activations.items():
            if len(tensor.shape) == 4:
                tensor = tensor[0]  # 1,channel,H,W =>channel,H,W
            tensor = tensor.unsqueeze(1)  # channel,1,H,W

            self.watcher.image(tensor, grid_size=grid_size, tag=layer_name, global_step=global_step, shuffle=shuffle)

    @staticmethod
    def _build_dir(dirs: str):
        if not os.path.exists(dirs):
            os.makedirs(dirs)


if __name__ == '__main__':
    from torch.nn import Conv2d


    class CL(Module):
        def __init__(self):
            super(CL, self).__init__()
            self.unet_block_1 = Conv2d(1, 2, 3, 1, 1)
            self.unet_block_2 = Conv2d(2, 4, 3, 1, 1)
            self.unet_block_3 = Conv2d(4, 8, 3, 1, 1)
            self.unet_block_4 = Conv2d(8, 16, 3, 1, 1)

        def forward(self, input):
            out = self.unet_block_1(input)
            out = self.unet_block_2(out)
            out = self.unet_block_3(out)
            out = self.unet_block_4(out)
            return out


    _model = CL()
    fv = FeatureVisualization(_model, "incep_test")
    _input = torch.randn(1, 1, 10, 10)
    fv.show_feature_trace(_input, ["unet_block_1", "unet_block_2", "unet_block_3"], 1)
    fv.watcher.close()
