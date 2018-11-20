from .sup_gan import SupGanTrainer
from abc import abstractmethod
import torch
from jdit.optimizer import Optimizer
from jdit.model import Model
from jdit.dataset import DataLoadersFactory

class Pix2pixGanTrainer(SupGanTrainer):
    d_turn = 1

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        """ A pixel to pixel gan trainer

        :param logdir:
        :param nepochs:
        :param gpu_ids_abs:
        :param netG:
        :param netD:
        :param optG:
        :param optD:
        :param datasets:
        :param latent_shape:
        """
        super(Pix2pixGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets)
        # self.loger.regist_config(self)

    def get_data_from_loader(self, batch_data):

        input_cpu, ground_truth_cpu = batch_data[0], batch_data[1]
        return input_cpu.to(self.device), ground_truth_cpu.to(self.device)

    def _watch_images(self, tag, grid_size=(3, 3), shuffle=False, save_file=True):
        self.watcher.image(self.input,
                           self.current_epoch,
                           tag="%s/input" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.fake,
                           self.current_epoch,
                           tag="%s/fake" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.ground_truth,
                           self.current_epoch,
                           tag="%s/real" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

    @abstractmethod
    def compute_d_loss(self):
        """ Rewrite this method to compute your own loss discriminator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake.detach())
            d_real = self.netD(self.ground_truth)
            var_dic = {}
            var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input,
                                             use_gpu=self.use_gpu)
            var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
            var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp
            return: loss_d, var_dic

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

            d_fake = self.netD(self.fake)
            var_dic = {}
            var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
            var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc
            return loss_g, var_dic

        """
        loss_g = None
        var_dic = {}
        return loss_g, var_dic

    @abstractmethod
    def compute_valid(self):
        """Rewrite this method to compute valid_epoch values.

        You can return a ``dict`` of values that you want to visualize.

        .. note::

            This method is under ``torch.no_grad():``. So, it will never compute grad.
            If you want to compute grad, please use ``torch.enable_grad():`` to wrap your operations.

        Example::

            d_fake = self.netD(self.fake.detach())
            d_real = self.netD(self.ground_truth)
            var_dic = {}
            var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
            return var_dic

        """

        var_dic = {}
        return var_dic

    def valid_epoch(self):
        super(Pix2pixGanTrainer, self).valid_epoch()
        self.netG.eval()
        self.netD.eval()
        if self.fixed_input is None:
            for iteration, batch in enumerate(self.datasets.loader_test, 1):
                if isinstance(batch,list):
                    self.fixed_input, fixed_ground_truth = self.get_data_from_loader(batch)
                    self.watcher.image(self.fixed_input, self.current_epoch, tag="Fixed/groundtruth",
                                       grid_size=(6, 6),
                                       shuffle=False)
                else:
                    self.fixed_input = batch.to(self.device)
                self.watcher.image(self.fixed_input, self.current_epoch, tag="Fixed/input",
                                   grid_size=(6, 6),
                                   shuffle=False)
                break
        # watching the variation during training by a fixed input
        with torch.no_grad():
            fake = self.netG(self.fixed_input).detach()
        self.watcher.image(fake, self.current_epoch, tag="Fixed/fake", grid_size=(6, 6), shuffle=False)

        # saving training processes to build a .gif.
        self.watcher.set_training_progress_images(fake, grid_size=(6, 6))

        self.netG.train()
        self.netD.train()

    def test(self):
        """ Test your model when you finish all epochs.

        This method will call when all epochs finish.

        Example::

            for index, batch in enumerate(self.datasets.loader_test, 1):
                # For test only have input without groundtruth
                input = batch.to(self.device)
                self.netG.eval()
                with torch.no_grad():
                    fake = self.netG(input)
                self.watcher.image(fake, self.current_epoch, tag="Test/fake", grid_size=(4, 4), shuffle=False)
            self.netG.train()
        """
        for index, batch in enumerate(self.datasets.loader_test, 1):
            # For test only have input without groundtruth
            self.input = batch.to(self.device)
            self.netG.eval()
            with torch.no_grad():
                fake = self.netG(self.input).detach()
            self.watcher.image(fake, self.current_epoch, tag="Test/fake", grid_size=(7, 7), shuffle=False)
        self.netG.train()

    # @property
    # def configure(self):
    #     dict = super(Pix2pixGanTrainer, self).configure
    #     dict["d_turn"] = self.d_turn
    #     return dict
