from ..super import SupTrainer
from tqdm import tqdm
import torch
from jdit.optimizer import Optimizer
from jdit.model import Model
from jdit.dataset import DataLoadersFactory


class SupSingleModelTrainer(SupTrainer):
    """ This is a Single Model Trainer.
    It means you only have one model.
        input, gound_truth
        output = model(input)
        loss(output, gound_truth)

    """

    def __init__(self, logdir, nepochs, gpu_ids_abs, net: Model, opt: Optimizer, datasets: DataLoadersFactory):

        super(SupSingleModelTrainer, self).__init__(nepochs, logdir, gpu_ids_abs=gpu_ids_abs)
        self.net = net
        self.opt = opt
        self.datasets = datasets
        self.fixed_input = None
        self.input = None
        self.output = None
        self.ground_truth = None

    def train_epoch(self, subbar_disable=False):
        for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1), unit="step", disable=subbar_disable):
            self.step += 1
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            self.output = self.net(self.input)
            self._train_iteration(self.opt, self.compute_loss, csv_filename="Train")
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
            self.watcher.image(self.output,
                           self.current_epoch,
                           tag="%s/output" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

            # show ground_truth
            self.watcher.image(self.ground_truth,
                           self.current_epoch,
                           tag="%s/ground_truth" % tag,
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
        self.watcher.image(self.output,
                           self.current_epoch,
                           tag="%s/output" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.ground_truth,
                           self.current_epoch,
                           tag="%s/ground_truth" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

    def compute_loss(self) -> (torch.Tensor, dict):
        """ Rewrite this method to compute your own loss Discriminator.
        Use self.input, self.output and self.ground_truth to compute loss.
        You should return a **loss** for the first position.
        You can return a ``dict`` of loss that you want to visualize on the second position.like

        Example::

            var_dic = {}
            var_dic["LOSS"] = loss_d = (self.output ** 2 - self.groundtruth ** 2) ** 0.5
            return: loss, var_dic

        """
        loss: torch.Tensor
        var_dic = {}

        return loss, var_dic

    def compute_valid(self) -> dict:
        """ Rewrite this method to compute your validation values.
        Use self.input, self.output and self.ground_truth to compute valid loss.
        You can return a ``dict`` of validation values that you want to visualize.

        Example::

            # It will do the same thing as ``compute_loss()``
            var_dic, _ = self.compute_loss()
            return var_dic

        """
        # It will do the same thing as ``compute_loss()``
        var_dic, _ = self.compute_loss()
        return var_dic

    def valid_epoch(self):
        """Validate model each epoch.

        It will be called each epoch, when training finish.
        So, do same verification here.

        Example::

        avg_dic: dict = {}
        self.net.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            with torch.no_grad():
                self.output = self.net(self.input)
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
        self.loger.write(self.step, self.current_epoch, avg_dic, "Valid", header=self.step <= 1)
        self._watch_images(tag="Valid")
        self.net.train()

        """
        avg_dic: dict = {}
        self.net.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            with torch.no_grad():
                self.output = self.net(self.input)
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
        self.net.train()

    def test(self):
        pass
