from ..super import SupTrainer
from tqdm import tqdm
import torch
from jdit.optimizer import Optimizer
from jdit.model import Model
from jdit.dataset import DataLoadersFactory


class SupGanTrainer(SupTrainer):
    d_turn = 1
    """The training times of Discriminator every ones Generator training.
    """

    def __init__(self, logdir, nepochs, gpu_ids_abs, netG: Model, netD: Model, optG: Optimizer, optD: Optimizer,
                 datasets: DataLoadersFactory):

        super(SupGanTrainer, self).__init__(nepochs, logdir, gpu_ids_abs=gpu_ids_abs)
        self.netG = netG
        self.netD = netD
        self.optG = optG
        self.optD = optD
        self.datasets = datasets
        self.fake = None
        self.fixed_input = None

    def train_epoch(self, subbar_disable=False):
        for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1), unit="step", disable=subbar_disable):
            self.step += 1
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            self.fake = self.netG(self.input)
            self._train_iteration(self.optD, self.compute_d_loss, csv_filename="Train_D")
            if (self.step % self.d_turn) == 0:
                self._train_iteration(self.optG, self.compute_g_loss, csv_filename="Train_G")
            if iteration == 1:
                self._watch_images("Train")

    def get_data_from_batch(self, batch_data: list, device: torch.device):
        """ Load and wrap data from the data lodaer.

            Split your one batch data to specify variable.

            Example::

                # batch_data like this [input_Data, ground_truth_Data]
                input_cpu, ground_truth_cpu = batch_data[0], batch_data[1]
                # then move them to device and return them
                return input_cpu.to(self.device), ground_truth_cpu.to(self.device)

        :param batch_data: one batch data load from ``DataLoader``
        :param device: A device variable. ``torch.device``
        :return: input Tensor, ground_truth Tensor
        """
        input_tensor, ground_truth_tensor = batch_data[0], batch_data[1]
        return input_tensor, ground_truth_tensor

    def _watch_images(self, tag: str, grid_size: tuple = (3, 3), shuffle=False, save_file=True):
        """ Show images in tensorboard

        To show images in tensorboad. If want to show fixed input and it's output,
        please use ``shuffle=False`` to fix the visualized data.
        Otherwise, it will sample and visualize the data randomly.

        Example::

            # show fake data
            self.watcher.image(self.fake,
                           self.current_epoch,
                           tag="%s/fake" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

            # show ground_truth
            self.watcher.image(self.ground_truth,
                           self.current_epoch,
                           tag="%s/real" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

            # show input
            self.watcher.image(self.input,
                           self.current_epoch,
                           tag="%s/input" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)


        :param tag: tensorboard tag
        :param grid_size: A tuple for grad size which data you want to visualize
        :param shuffle: If shuffle the data.
        :param save_file: If save this images.
        :return:
        """
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

    def compute_d_loss(self) -> (torch.Tensor, dict):
        """ Rewrite this method to compute your own loss Discriminator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake.detach())
            d_real = self.netD(self.ground_truth)
            var_dic = {}
            var_dic["GP"] = gp = gradPenalty(self.netD, self.ground_truth, self.fake, input=self.input,
                                             use_gpu=self.use_gpu)
            var_dic["WD"] = w_distance = (d_real.mean() - d_fake.mean()).detach()
            var_dic["LOSS_D"] = loss_d = d_fake.mean() - d_real.mean() + gp + sgp
            return: loss_d, var_dic

        """
        loss_d: torch.Tensor
        var_dic = {}

        return loss_d, var_dic

    def compute_g_loss(self) -> (torch.Tensor, dict):
        """Rewrite this method to compute your own loss of Generator.

        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            d_fake = self.netD(self.fake)
            var_dic = {}
            var_dic["JC"] = jc = jcbClamp(self.netG, self.input, use_gpu=self.use_gpu)
            var_dic["LOSS_D"] = loss_g = -d_fake.mean() + jc
            return: loss_g, var_dic

        """
        loss_g: torch.Tensor
        var_dic = {}
        return loss_g, var_dic

    def compute_valid(self) -> dict:
        """ Rewrite this method to compute your validation values.

        You can return a ``dict`` of validation values that you want to visualize.

        Example::

            # It will do the same thing as ``compute_g_loss()`` and ``self.compute_d_loss()``
            g_loss, _ = self.compute_g_loss()
            d_loss, _ = self.compute_d_loss()
            var_dic = {"LOSS_D": d_loss, "LOSS_G": g_loss}
            return var_dic

        """
        g_loss, _ = self.compute_g_loss()
        d_loss, _ = self.compute_d_loss()
        var_dic = {"LOSS_D": d_loss, "LOSS_G": g_loss}
        return var_dic

    def valid_epoch(self):
        """Validate model each epoch.

        It will be called each epoch, when training finish.
        So, do same verification here.

        Example::

            avg_dic: dict = {}
            self.netG.eval()
            self.netD.eval()
            # Load data from loader_valid.
            for iteration, batch in enumerate(self.datasets.loader_valid, 1):
                self.input, self.ground_truth = self.get_data_from_batch(batch)
                with torch.no_grad():
                    self.fake = self.netG(self.input)
                    # You can write this function to apply your computation.
                    dic: dict = self.compute_valid()
                if avg_dic == {}:
                    avg_dic: dict = dic
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

        """
        avg_dic: dict = {}
        self.netG.eval()
        self.netD.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            with torch.no_grad():
                self.fake = self.netG(self.input)
                dic: dict = self.compute_valid()
            if avg_dic == {}:
                avg_dic: dict = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(avg_dic, self.step, tag="Valid")
        self.loger.write(self.step, self.current_epoch, avg_dic, "Valid", header=self.current_epoch <= 1)
        self._watch_images(tag="Valid")
        self.netG.train()
        self.netD.train()

    def test(self):
        pass

    @property
    def configure(self):
        config_dic = super(SupGanTrainer, self).configure
        config_dic["d_turn"] = str(self.d_turn)
        return config_dic
