import pandas as pd
import torch
import numpy as np
import os
from scipy import linalg
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
from .fid import InceptionV3


# class InceptionV3(nn.Module):
#     """Pretrained _InceptionV3 network returning feature maps"""
#
#     # Index of default block of inception to return,
#     # corresponds to output of final average pooling
#     DEFAULT_BLOCK_INDEX = 3
#
#     # Maps feature dimensionality to their output blocks indices
#     BLOCK_INDEX_BY_DIM = {
#         64: 0,  # First max pooling features
#         192: 1,  # Second max pooling featurs
#         768: 2,  # Pre-aux classifier features
#         2048: 3  # Final average pooling features
#     }
#
#     def __init__(self,
#                  output_blocks=[DEFAULT_BLOCK_INDEX],
#                  resize_input=True,
#                  normalize_input=True,
#                  requires_grad=False):
#         """Build pretrained _InceptionV3
#
#         Parameters
#         ----------
#         output_blocks : list of int
#             Indices of blocks to return features of. Possible values are:
#                 - 0: corresponds to output of first max pooling
#                 - 1: corresponds to output of second max pooling
#                 - 2: corresponds to output which is fed to aux classifier
#                 - 3: corresponds to output of final average pooling
#         resize_input : bool
#             If true, bilinearly resizes input to width and height 299 before
#             feeding input to model. As the network without fully connected
#             layers is fully convolutional, it should be able to handle inputs
#             of arbitrary size, so resizing might not be strictly needed
#         normalize_input : bool
#             If true, normalizes the input to the statistics the pretrained
#             Inception network expects
#         requires_grad : bool
#             If true, parameters of the model require gradient. Possibly useful
#             for finetuning the network
#         """
#         super(InceptionV3, self).__init__()
#
#         self.resize_input = resize_input
#         self.normalize_input = normalize_input
#         self.output_blocks = sorted(output_blocks)
#         self.last_needed_block = max(output_blocks)
#
#         assert self.last_needed_block <= 3, \
#             'Last possible output block index is 3'
#
#         self.blocks = nn.ModuleList()
#
#         inception = models.inception_v3(pretrained=True)
#
#         # Block 0: input to maxpool1
#         block0 = [
#             inception.Conv2d_1a_3x3,
#             inception.Conv2d_2a_3x3,
#             inception.Conv2d_2b_3x3,
#             nn.MaxPool2d(kernel_size=3, stride=2)
#         ]
#         self.blocks.append(nn.Sequential(*block0))
#
#         # Block 1: maxpool1 to maxpool2
#         if self.last_needed_block >= 1:
#             block1 = [
#                 inception.Conv2d_3b_1x1,
#                 inception.Conv2d_4a_3x3,
#                 nn.MaxPool2d(kernel_size=3, stride=2)
#             ]
#             self.blocks.append(nn.Sequential(*block1))
#
#         # Block 2: maxpool2 to aux classifier
#         if self.last_needed_block >= 2:
#             block2 = [
#                 inception.Mixed_5b,
#                 inception.Mixed_5c,
#                 inception.Mixed_5d,
#                 inception.Mixed_6a,
#                 inception.Mixed_6b,
#                 inception.Mixed_6c,
#                 inception.Mixed_6d,
#                 inception.Mixed_6e,
#             ]
#             self.blocks.append(nn.Sequential(*block2))
#
#         # Block 3: aux classifier to final avgpool
#         if self.last_needed_block >= 3:
#             block3 = [
#                 inception.Mixed_7a,
#                 inception.Mixed_7b,
#                 inception.Mixed_7c,
#                 nn.AdaptiveAvgPool2d(output_size=(1, 1))
#             ]
#             self.blocks.append(nn.Sequential(*block3))
#
#         for param in self.parameters():
#             param.requires_grad = requires_grad
#
#     def forward(self, inp):
#         """Get Inception feature maps
#
#         Parameters
#         ----------
#         inp : torch.autograd.Variable
#             Input tensor of shape Bx3xHxW. Values are expected to be in
#             range (0, 1)
#
#         Returns
#         -------
#         List of torch.autograd.Variable, corresponding to the selected output
#         block, sorted ascending by index
#         """
#         outp = []
#         x = inp
#
#         if self.resize_input:
#             # x = F.upsample(x, size=(299, 299), mode='bilinear')
#             x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
#         if self.normalize_input:
#             x = x.clone()
#             x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
#             x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
#             x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
#
#         for idx, block in enumerate(self.blocks):
#             x = block(x)
#             if idx in self.output_blocks:
#                 outp.append(x)
#
#             if idx == self.last_needed_block:
#                 break
#
#         return outp


class FID(object):
    def __init__(self, gpu_ids, dim=2048, ):
        self.inception = None
        self.dim = dim
        self.gpu_ids = gpu_ids

        self.mu = None
        self.sigma = None

    def _get_cifar10_mu_sigma(self):
        if self.mu is None:
            self.mu = pd.read_csv(os.path.join(os.getcwd(), "jdit/metric/Cifar10_M.csv"), header=None).values
            self.mu = self.mu.reshape(-1)
        if self.sigma is None:
            self.sigma = pd.read_csv(os.path.join(os.getcwd(), "jdit/metric/Cifar10_S.csv"), header=None).values
        return self.mu, self.sigma

    def _getInception(self):
        if self.inception is None:
            self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[self.dim]])
            self.inception = self.inception.cuda() if len(self.gpu_ids) > 0 else self.inception
        return self.inception

    # def cifar10_FID_fromloader(self, dataloader):
    #
    #     fake_mu, fake_sigma = self.compute_act_statistics_from_loader(dataloader, self._getInception(), self.gpu_ids)
    #     cifar10_mu, cifar10_sigma = self._get_cifar10_mu_sigma()
    #     fid = self.FID(cifar10_mu, cifar10_sigma, fake_mu, fake_sigma)
    #     return fid
    #
    # def compute_act_statistics_from_loader(self, dataloader, model, gpu_ids):
    #     model.eval()
    #     pred_arr = None
    #     placeholder = Variable().cuda() if len(gpu_ids) > 0 else Variable()
    #     model = model.cuda() if len(gpu_ids) > 0 else model
    #     for iteration, batch in enumerate(dataloader, 1):
    #         pred = self.compute_act_batch(placeholder, batch)
    #         if pred_arr is None:
    #             pred_arr = pred
    #         else:
    #             pred_arr = torch.cat((pred_arr, pred))  # [?, 2048, 1, 1]
    #
    #     pred_arr = pred_arr.cpu().numpy().reshape(pred_arr.size()[0], -1)  # [?, 2048]
    #     mu = np.mean(pred_arr, axis=0)
    #     sigma = np.cov(pred_arr, rowvar=False)
    #     return mu, sigma

    def compute_act_batch(self, batch):
        with torch.autograd.no_grad():
            pred = self.inception(batch)[0]  # [batchsize, 1024,1,1]
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred

    def evaluate_model_fid(self, model, noise_shape=(64, 3, 32, 32), amount=10000, dataset="cifar10"):
        model.eval()
        noise = Variable().cuda() if len(self.gpu_ids) > 0 else Variable()
        # iteration = amount // noise_shape[0]
        iteration = amount
        pred_arr = None
        self._getInception()
        with torch.autograd.no_grad():
            for i in range(iteration):
                noise_cpu = Variable(torch.randn(noise_shape))
                noise.data.resize_(noise_cpu.size()).copy_(noise_cpu)
                fake = model(noise)
                pred = self.inception(fake)[0]  # [batchsize, 1024,1,1]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        if pred_arr is None:
            pred_arr = pred
        else:
            pred_arr = torch.cat((pred_arr, pred))  # [?, 2048, 1, 1]
        pred_arr = pred_arr.cpu().numpy().reshape(pred_arr.size()[0], -1)  # [?, 2048]
        fake_mu = np.mean(pred_arr, axis=0)
        fake_sigma = np.cov(pred_arr, rowvar=False)
        if dataset == "cifar10":
            self._get_cifar10_mu_sigma()
        # print(fake_mu.shape, fake_sigma.shape)
        # print(self.mu.shape, self.sigma.shape)
        fid_score = self.FID(fake_mu, fake_sigma, self.mu, self.sigma)
        model.train()
        return fid_score

    @staticmethod
    def FID(mu1, sigma1, mu2, sigma2, eps=1e-6):
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

        if mu1.shape != mu2.shape:
            raise ValueError("Training and test mean vectors have different lengths")
        if sigma1.shape != sigma2.shape:
            raise ValueError("Training and test covariances have different dimensions")

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
