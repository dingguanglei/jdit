# coding=utf-8
import torch
from math import log10
from torch.nn import Parameter, Module, Linear, MSELoss
from torch.autograd import Variable, grad
import numpy as np
from PIL.ImageEnhance import *


def gradPenalty(D_net, real, fake, LAMBDA=10, input=None, use_gpu=False):
    batch_size = real.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real)

    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * real + ((1 - alpha) * fake)

    if use_gpu:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    if input is not None:
        disc_interpolates = D_net(interpolates, input)
    else:
        disc_interpolates = D_net(interpolates)
    gradients = grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda()
        if use_gpu else torch.ones(disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def Branch_gradPenalty(D_net, reals, fakes, LAMBDA=10, input=None, use_gpu=False):
    batch_size = reals.size()[0]
    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(reals)

    alpha = alpha.cuda() if use_gpu else alpha

    interpolates = alpha * reals + ((1 - alpha) * fakes)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    if input is not None:
        shape = fakes.shape
        fake_NN = fakes[:, 0, :, :].view(shape[0], 1, shape[2], shape[3])
        fake_NN_NBG_SR = fakes[:, 1, :, :].view(shape[0], 1, shape[2], shape[3])
        fake_GAUSSIAN = fakes[:, 2, :, :].view(shape[0], 1, shape[2], shape[3])
        d_fake_NN = D_net(fake_NN, input)[:, 0, :, :].view(shape[0], 1, 16, 16)
        d_fake_NN_NBG_SR = D_net(fake_NN_NBG_SR, input)[:, 1, :, :].view(shape[0], 1, 16, 16)
        d_fake_GAUSSIAN = D_net(fake_GAUSSIAN, input)[:, 2, :, :].view(shape[0], 1, 16, 16)
        disc_interpolates = torch.cat([d_fake_NN, d_fake_NN_NBG_SR, d_fake_GAUSSIAN], 1)
    else:
        disc_interpolates = D_net(interpolates)
    gradients = grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda()
        if use_gpu else torch.ones(disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def spgradPenalty(D_net, real, fake, input=None, type="G", use_gpu=False):
    """sample gradient penalty

    :param D_net:
    :param real:
    :param fake:
    :param input: only for CGAN, net_D(fake,input)
    :param type: select "G" or "X". default "X"
    :return:
    """

    if type == "G":
        data = fake
    elif type == "X":
        data = real

    data = Variable(data, requires_grad=True)
    if input is not None:
        d = D_net(data, input)
    else:
        d = D_net(data)
    gradients = grad(
        outputs=d,
        inputs=data,
        grad_outputs=torch.ones(d.size()).cuda()
        if use_gpu else torch.ones(d.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    sgp = (gradients.norm(2, dim=1) ** 2).mean()
    return sgp


def jcbClamp(G_net, z, lmbda_max=20, lmbda_min=1, ep=1, use_gpu=False):
    """ implement of jacobin climping.
    'Is Generator Conditioning Causally Related to GAN Performance?'
    :param G_net: generate model
    :param z: input
    :param lmbda_max: default 20
    :param lmbda_min: default 1
    :param ep: default 1
    :param use_gpu: default False
    :return:
    """
    lmbda_max = lmbda_max * torch.ones(1)
    lmbda_min = lmbda_min * torch.ones(1)
    sigma = torch.randn(z.size())
    if use_gpu:
        lmbda_max = lmbda_max.cuda()
        lmbda_min = lmbda_min.cuda()
        sigma = sigma.cuda()
    sigma = sigma / torch.norm(sigma, 2) * ep
    z_ = z + sigma
    Q = torch.norm(G_net(z) - G_net(z_), 2) / torch.norm(z - z_, 2)
    l_max = (torch.max(Q, lmbda_max) - lmbda_max) ** 2
    l_min = (torch.min(Q, lmbda_min) - lmbda_min) ** 2
    return (l_max + l_min).mean()


def getPsnr(fake, real, use_gpu=False):
    mse_loss = MSELoss()
    if use_gpu:
        mse_loss = mse_loss.cuda()
    mse = mse_loss(fake, real)
    psnr = 10 * log10(1 / mse.data.item())
    return psnr


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length, mask_numb=.0):
        self.n_holes = n_holes
        self.length = length
        self.mask_numb = mask_numb

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = self.mask_numb

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img



class SpectralNorm(Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
