from .sup_gan import SupGanTrainer
from abc import abstractmethod
from torch.autograd import Variable
import torch
from jdit.optimizer import Optimizer
from jdit.model import Model
from jdit.dataset import DataLoadersFactory


class GenerateGanTrainer(SupGanTrainer):
    d_turn = 1

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets, latent_shape):
        """ a gan super class

        :param logdir:Path of log
        :param nepochs:Amount of epochs.
        :param gpu_ids_abs: he id of gpus which t obe used. If use CPU, set ``[]``.
        :param netG:Generator model.
        :param netD:Discrimiator model
        :param optG:Optimizer of Generator.
        :param optD:Optimizer of Discrimiator.
        :param datasets:Datasets.
        :param latent_shape:The shape of input noise.
        """
        super(GenerateGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets)
        self.latent_shape = latent_shape
        # self.metric = FID(self.gpu_ids)

    def get_data_from_loader(self, batch_data):
        ground_truth_cpu = batch_data[0]
        input_cpu = Variable(torch.randn((len(ground_truth_cpu), *self.latent_shape)))
        return input_cpu.to(self.device), ground_truth_cpu.to(self.device)

    def valid_epoch(self):
        avg_dic = dict()
        self.netG.eval()
        self.netD.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_loader(batch)
            self.fake = self.netG(self.input)
            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(avg_dic, self.step, tag="Valid")
        self._watch_images(tag="Valid")
        self.netG.train()
        self.netD.train()

    @abstractmethod
    def compute_d_loss(self):
        """ Rewrite this method to compute your own loss discriminator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake.detach())
            d_real = self.netD(self.ground_truth)
            var_dic = {}
            var_dic["LS_LOSSD"] = loss_d = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
            return loss_d, var_dic

        """
        loss_d = None
        var_dic = {}

        return loss_d, var_dic

    @abstractmethod
    def compute_g_loss(self):
        """Rewrite this method to compute your own loss of generator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake, self.input)
            var_dic = {}
            var_dic["LS_LOSSG"] = loss_g = 0.5 * torch.mean((d_fake - 1) ** 2)
            return loss_g, var_dic

        """
        loss_g = None
        var_dic = {}
        return loss_g, var_dic

    @abstractmethod
    def compute_valid(self):
        _, d_var_dic = self.compute_g_loss()
        _, g_var_dic = self.compute_d_loss()
        var_dic = dict(d_var_dic, **g_var_dic)
        return var_dic

    def test(self):
        self.input = Variable(torch.randn((16, *self.latent_shape))).to(self.device)
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(self.input).detach()
        self.watcher.image(fake, self.current_epoch, tag="Test/fake", grid_size=(4, 4), shuffle=False)
        self.netG.train()

    @property
    def configure(self):
        config_dic = super(GenerateGanTrainer, self).configure
        config_dic["latent_shape"] = str(self.latent_shape)
        return config_dic
