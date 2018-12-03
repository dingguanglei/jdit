import torch
import os
from typing import Union
import numpy as np
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from jdit.model import Model
import random


class Watcher(object):
    """this is a params and images watcher

    """

    def __init__(self, logdir: str, mode: str = "L"):
        self.logdir = logdir
        self.writer = SummaryWriter(log_dir=logdir)
        self.mode = mode
        self._build_dir(logdir)
        self.training_progress_images = []
        self.gif_duration = 0.5

    def model_params(self, model: torch.nn.Module, global_step: int):
        for name, param in model.named_parameters():
            if "bias" in name:
                continue
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step)

    def scalars(self, var_dict: dict, global_step: int, tag="Train"):
        for key, scalar in var_dict.items():
            self.writer.add_scalars(key, {tag: scalar}, global_step)

    def _sample(self, tensor: torch.Tensor, num_samples: int, shuffle=True):
        total = len(tensor)
        assert num_samples <= total
        if shuffle:
            rand_index = random.sample(list(range(total)), num_samples)
            sampled_tensor: torch.Tensor = tensor[rand_index]
        else:
            sampled_tensor: torch.Tensor = tensor[:num_samples]
        return sampled_tensor

    def image(self, img_tensors: torch.Tensor, global_step: int, tag: str = "Train/input",
              grid_size: Union[list, tuple] = (3, 1), shuffle=True, save_file=False):
        # if input is (-1, 1).this should be -1. to convert to (0,1)   mean=(-1, -1, -1), std=(2, 2, 2)
        assert len(img_tensors.size()) == 4, "img_tensors rank should be 4, got %d instead" % len(img_tensors.size())
        self._build_dir(os.path.join(self.logdir, "plots", tag))
        rows, columns = grid_size[0], grid_size[1]
        batch_size = len(img_tensors)  # img_tensors =>(batchsize, 3, 256, 256)
        num_samples: int = min(batch_size, rows * columns)
        assert len(img_tensors) >= num_samples, "you want to show grid %s, but only have %d tensors to show." % (
            grid_size, len(img_tensors))

        sampled_tensor = self._sample(img_tensors, num_samples,
                                      shuffle).detach().cpu()  # (sample_num, 3, 32,32)  tensors
        # sampled_images = map(transforms.Normalize(mean, std), sampled_tensor)  # (sample_num, 3, 32,32) images
        sampled_images: torch.Tensor = make_grid(sampled_tensor, nrow=rows, normalize=True, scale_each=True)
        self.writer.add_image(tag, sampled_images, global_step)

        if save_file:
            img = transforms.ToPILImage()(sampled_images).convert(self.mode)
            filename = "%s/plots/%s/E%03d.png" % (self.logdir, tag, global_step)
            img.save(filename)

    def embedding(self, data: torch.Tensor, label_img: torch.Tensor = None, label=None, global_step: int = None,
                  tag: str = "embedding"):
        """ Show PCA, t-SNE of `mat` on tensorboard

        :param data: An img tensor with shape  of (N, C, H, W)
        :param label_img: Label img on each data point.
        :param label: Label of each img. It will convert to str.
        :param global_step: Img step label.
        :param tag: Tag of this plot.
        """
        features = data.view(len(data), -1)
        self.writer.add_embedding(features, metadata=label, label_img=label_img, global_step=global_step, tag=tag)

    def set_training_progress_images(self, img_tensors: torch.Tensor, grid_size: Union[list, tuple] = (3, 1)):
        assert len(img_tensors.size()) == 4, "img_tensors rank should be 4, got %d instead" % len(img_tensors.size())
        rows, columns = grid_size[0], grid_size[1]
        batch_size = len(img_tensors)  # img_tensors =>(batchsize, 3, 256, 256)
        num_samples = min(batch_size, rows * columns)
        assert len(img_tensors) >= num_samples, "you want to show grid %s, but only have %d tensors to show." % (
            grid_size, len(img_tensors))
        sampled_tensor = self._sample(img_tensors, num_samples, False).detach().cpu()  # (sample_num, 3, 32,32)  tensors
        sampled_images = make_grid(sampled_tensor, nrow=rows, normalize=True, scale_each=True)
        img_grid = np.transpose(sampled_images.numpy(), (1, 2, 0))
        self.training_progress_images.append(img_grid)

    def save_in_gif(self):
        import imageio, warnings
        filename = "%s/plots/training.gif" % (self.logdir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imageio.mimsave(filename, self.training_progress_images, duration=self.gif_duration)
        self.training_progress_images = None

    def graph(self, model: Union[torch.nn.Module, torch.nn.DataParallel, Model],
              input_shape: Union[list, tuple] = None, use_gpu=False,
              *input):
        if isinstance(model, torch.nn.Module):
            proto_model: torch.nn.Module = model
            num_params: int = self._count_params(proto_model)
        elif isinstance(model, torch.nn.DataParallel):
            proto_model: torch.nn.Module = model.module
            num_params: int = self._count_params(proto_model)
        elif isinstance(model, Model):
            proto_model: torch.nn.Module = model.model
            num_params: int = model.num_params
        else:
            raise TypeError("Only `nn.Module`, `nn.DataParallel` and `Model` can be passed!")

        if input_shape is not None:
            assert (isinstance(input_shape, tuple) or isinstance(input_shape, list)), \
                "param 'input_shape' should be list or tuple."
            input_tensor = torch.ones(input_shape).cuda() if use_gpu else torch.ones(input_shape)
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

            self.scalars({'ParamsNum': num_params}, 0, tag="ParamsNum")
            res = proto_model(input_tensor)
            self.scalars({'ParamsNum': num_params}, 1, tag="ParamsNum")
            del res
            self.writer.add_graph(model, input_tensor)
        else:
            self.scalars({'ParamsNum': num_params}, 0, tag="ParamsNum")
            res = proto_model(*input)
            self.scalars({'ParamsNum': num_params}, 1, tag="ParamsNum")
            del res
            self.writer.add_graph(proto_model, input)

    def _count_params(self, proto_model: torch.nn.Module):
        """count the total parameters of model.

        :param proto_model: pytorch module
        :return: number of parameters
        """
        num_params = 0
        for param in proto_model.parameters():
            num_params += param.numel()
        return num_params

    def close(self):
        # self.writer.export_scalars_to_json("%s/scalers.json" % self.logdir)
        if self.training_progress_images:
            self.save_in_gif()
        self.writer.close()

    def _build_dir(self, dirs: str):
        if not os.path.exists(dirs):
            # print("%s directory is not found. Build now!" % dir)
            os.makedirs(dirs)

# activation = {}
class FeatureVisualization():
    def __init__(self, model):
        model.eval()
        self.model = model
        self.activation = {}

    def _hook(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()

        return hook

    def _register_forward_hook(self):
        self.model.unet_block_1.register_forward_hook(self._hook('block1'))
        self.model.unet_block_2.register_forward_hook(self._hook('block2'))
        self.model.unet_block_3.register_forward_hook(self._hook('block3'))

    def trace_activation(self, input):
        self._register_forward_hook()
        output = self.model(input)
        return output

class CL(torch.nn.Module):
    def __init__(self):
        super(CL, self).__init__()
        self.unet_block_1 = torch.nn.Conv2d(1, 2, 3, 1, 1)
        self.unet_block_2 = torch.nn.Conv2d(2, 4, 3, 1, 1)
        self.unet_block_3 = torch.nn.Conv2d(4, 8, 3, 1, 1)
        self.unet_block_4 = torch.nn.Conv2d(8, 16, 3, 1, 1)

    def forward(self, input):
        out = self.unet_block_1(input)
        out = self.unet_block_2(out)
        out = self.unet_block_3(out)
        out = self.unet_block_4(out)
        return out

if __name__ == '__main__':


    fv = FeatureVisualization(CL())
    print(fv.model.named_parameters())
    # def _register_forward_hook(self):
    #     self.model.unet_block_1.register_forward_hook(self._hook('block1'))
    #     self.model.unet_block_2.register_forward_hook(self._hook('block2'))
    #     self.model.unet_block_3.register_forward_hook(self._hook('block3'))
    # fv._register_forward_hook = _register_forward_hook

    fv.trace_activation()
    print(fv.activation.keys())

