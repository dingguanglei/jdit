from .sup_gan import SupGanTrainer
from abc import abstractmethod
import torch


class Pix2pixGanTrainer(SupGanTrainer):
    d_turn = 1

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        """ A pixel to pixel gan trainer

        :param logdir:Path of log
        :param nepochs:Amount of epochs.
        :param gpu_ids_abs: he id of gpus which t obe used. If use CPU, set ``[]``.
        :param netG:Generator model.
        :param netD:Discrimiator model
        :param optG:Optimizer of Generator.
        :param optD:Optimizer of Discrimiator.
        :param datasets:Datasets.
        """
        super(Pix2pixGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets)

    def get_data_from_batch(self, batch_data: list, device: torch.device):
        input_tensor, ground_truth_tensor = batch_data[0], batch_data[1]
        return input_tensor, ground_truth_tensor

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
        """ Rewrite this method to compute your own loss Discriminator.

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
        """Rewrite this method to compute your own loss of Generator.

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
        _, d_var_dic = self.compute_g_loss()
        _, g_var_dic = self.compute_d_loss()
        var_dic = dict(d_var_dic, **g_var_dic)
        return var_dic

    def valid_epoch(self):
        super(Pix2pixGanTrainer, self).valid_epoch()
        self.netG.eval()
        self.netD.eval()
        if self.fixed_input is None:
            for batch in self.datasets.loader_test:
                if isinstance(batch, (list, tuple)):
                    self.fixed_input, fixed_ground_truth = self.get_data_from_batch(batch, self.device)
                    self.watcher.image(fixed_ground_truth, self.current_epoch, tag="Fixed/groundtruth",
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
        for batch in self.datasets.loader_test:
            self.input, _ = self.get_data_from_batch(batch, self.device)
            self.netG.eval()
            with torch.no_grad():
                fake = self.netG(self.input).detach()
            self.watcher.image(fake, self.current_epoch, tag="Test/fake", grid_size=(7, 7), shuffle=False)
        self.netG.train()
