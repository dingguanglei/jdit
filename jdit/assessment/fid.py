#!/usr/bin/env python3
"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from tqdm import *
import math
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch
import numpy as np
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d


class InceptionV3(nn.Module):
    """Pretrained _InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
        }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        """Build pretrained _InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, normalizes the input to the statistics the pretrained
            Inception network expects
        requires_grad : bool
            If true, parameters of the model require gradient. Possibly useful
            for finetuning the network
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
            ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
                ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
                ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            # x = F.upsample(x, size=(299, 299), mode='bilinear')
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


# ______________________________________________________________

def compute_act_statistics_from_loader(dataloader: DataLoader, model, gpu_ids):
    """

    :param dataloader:
    :param model:
    :param gpu_ids:
    :return:
    """
    model.eval()
    pred_arr = None
    image = Variable().cuda() if len(gpu_ids) > 0 else Variable()
    model = model.cuda() if len(gpu_ids) > 0 else model
    for iteration, batch in tqdm(enumerate(dataloader, 1)):

        image.data.resize_(batch[0].size()).copy_(batch[0])
        with torch.autograd.no_grad():
            pred = model(image)[0]  # [batchsize, 1024,1,1]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        if pred_arr is None:
            pred_arr = pred
        else:
            pred_arr = torch.cat((pred_arr, pred))  # [?, 2048, 1, 1]

    pred_arr = pred_arr.cpu().numpy().reshape(pred_arr.size()[0], -1)  # [?, 2048]
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def compute_act_statistics_from_torch(tensor, model, gpu_ids):
    model.eval()
    model = model.cuda() if len(gpu_ids) > 0 else model

    image = Variable().cuda() if len(gpu_ids) > 0 else Variable()
    image.data.resize_(tensor.size()).copy_(tensor)  # [Total, C, H, W]
    with torch.autograd.no_grad():
        pred = model(image)[0]  # [batchsize, 1024,1,1]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    pred_np = pred.cpu().numpy().reshape(len(pred), -1)  # [?, 2048]
    mu = np.mean(pred_np, axis=0)
    sigma = np.cov(pred_np, rowvar=False)
    return mu, sigma


def compute_act_statistics_from_path(path, model, gpu_ids, dims, batch_size=32, verbose=True):
    if path.endswith('.npz'):
        f = np.load(path)
        mu, sigma = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        import pathlib
        from scipy.misc import imread
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

        images = np.array([imread(str(fn)).astype(np.float32) for fn in files])
        # Bring images to shape (B, 3, H, W)
        images = images.transpose((0, 3, 1, 2))
        # Rescale images to be between 0 and 1
        images /= 255

        # act = get_activations(images, model, batch_size, dims, cuda, verbose)
        model.eval()
        model = model.cuda() if len(gpu_ids) > 0 else model
        d0 = images.shape[0]
        if batch_size > d0:
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            batch_size = d0
        n_batches = d0 // batch_size
        n_used_imgs = n_batches * batch_size
        pred_arr = np.empty((n_used_imgs, dims))
        for i in range(n_batches):
            if verbose:
                print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                      end='', flush=True)
            start = i * batch_size
            end = start + batch_size

            batch = torch.from_numpy(images[start:end]).type(torch.FloatTensor)
            batch = Variable(batch, volatile=True)
            if len(gpu_ids) > 0:
                batch = batch.cuda()
            pred = model(batch)[0]
            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

        mu = np.mean(pred_arr, axis=0)
        sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representive data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representive data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


BLOCK_INDEX_BY_DIM = {
    64: 0,  # First max pooling features
    192: 1,  # Second max pooling featurs
    768: 2,  # Pre-aux classifier features
    2048: 3  # Final average pooling features
    }


def FID_score(source, target, sample_prop=1.0, gpu_ids=(), dim=2048, batchsize=128, verbose=True):
    """ Compute FID score from ``Tensor``, ``DataLoader`` or a directory``path``.


    :param source: source data.
    :param target: target data.
    :param sample_prop: If passing a ``Tensor`` source, set this rate to sample a part of data from source.
    :param gpu_ids: gpu ids.
    :param dim: The number of features. Three options available.

        * 64:   The first max pooling features of Inception.

        * 192:  The Second max pooling features of Inception.

        * 768:  The Pre-aux classifier features of Inception.

        * 2048: The Final average pooling features of Inception.

        Default: 2048.
    :param batchsize: Only using for passing paths of source and target.
    :param verbose: If show processing log.
    :return: fid score

    .. attention ::

        If you are passing ``Tensor`` as source and target.
        Make sure you have enough memory to load these data in _InceptionV3.
        Otherwise, please passing ``path`` of ``DataLoader`` to compute them step by step.

    Example::

        >>> from jdit.dataset import Cifar10
        >>> loader = Cifar10(root=r"../../datasets/cifar10", batch_shape=(32, 3, 32, 32))
        >>> target_tensor = loader.samples_train[0]
        >>> source_tensor = loader.samples_valid[0]
        >>> # using Tensor to compute FID score
        >>> fid_value = FID_score(source_tensor, target_tensor, sample_prop=0.01, depth=768)
        >>> print('FID: ', fid_value)
        >>> # using DataLoader to compute FID score
        >>> fid_value = FID_score(loader.loader_test, loader.loader_valid, depth=768)
        >>> print('FID: ', fid_value)

    """
    assert sample_prop <= 1 and sample_prop > 0, "sample_prop must between 0 and 1, but %s got" % sample_prop
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[dim]])
    if isinstance(source, Tensor) and isinstance(target, Tensor):
        # source[?,C,H,W] target[?,C,H,W]
        s_length = len(source)
        t_length = len(target)
        nums_sample = math.floor(min(s_length, t_length) * sample_prop)
        s_mu, s_sigma = compute_act_statistics_from_torch(source[0:nums_sample], model, gpu_ids)
        t_mu, t_sigma = compute_act_statistics_from_torch(target[0:nums_sample], model, gpu_ids)

    elif isinstance(source, DataLoader) and isinstance(target, DataLoader):
        s_mu, s_sigma = compute_act_statistics_from_loader(source, model, gpu_ids)
        t_mu, t_sigma = compute_act_statistics_from_loader(target, model, gpu_ids)
    elif isinstance(source, str) and isinstance(target, str):
        s_mu, s_sigma = compute_act_statistics_from_path(source, model, gpu_ids, dim, batchsize, verbose)
        t_mu, t_sigma = compute_act_statistics_from_path(target, model, gpu_ids, dim, batchsize, verbose)
    else:
        raise Exception

    fid_value = calculate_frechet_distance(s_mu, s_sigma, t_mu, t_sigma)
    return fid_value
