# from torch.nn import CrossEntropyLoss
from .sup_single import *
from abc import abstractmethod
# from tqdm import *


class ClassificationTrainer(SupSingleModelTrainer):
    """this is a classification trainer.

    """

    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets, num_class):
        super(ClassificationTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, datasets)
        self.net = net
        self.opt = opt
        self.datasets = datasets
        self.num_class = num_class


    @abstractmethod
    def compute_loss(self):
        """Compute the main loss and observed values.

        Compute the loss and other values shown in tensorboard scalars visualization.
        You should return a main loss for doing backward propagation.

        So, if you want some values visualized. Make a ``dict()`` with key name is the variable's name.
        The training logic is :
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            self.output = self.net(self.input)
            self._train_iteration(self.opt, self.compute_loss, csv_filename="Train")
        So, you have `self.net`, `self.input`, `self.output`, `self.ground_truth` to compute your own loss here.

        .. note::

          Only the main loss will do backward propagation, which is the first returned variable.
          If you have the joint loss, please add them up and return one main loss.

        .. note::

          All of your variables in returned ``dict()`` will never do backward propagation with ``model.train()``.
          However, It still compute grads, without using ``with torch.autograd.no_grad()``.
          So, you can compute any grads variables for visualization.

        Example::

          var_dic = {}
          labels = self.ground_truth.squeeze().long()
          var_dic["MSE"] = loss = nn.MSELoss()(self.output, labels)
          return loss, var_dic

        """

    @abstractmethod
    def compute_valid(self):
        """Compute the valid_epoch variables for visualization.

        Compute the validations.
        For the validations will only be used in tensorboard scalars visualization.
        So, if you want some variables visualized. Make a ``dict()`` with key name is the variable's name.
        You have `self.net`, `self.input`, `self.output`, `self.ground_truth` to compute your own validations here.

        .. note::

          All of your variables in returned ``dict()`` will never do backward propagation with ``model.eval()``.
          However, It still compute grads, without using ``with torch.autograd.no_grad()``.
          So, you can compute some grads variables for visualization.

        Example::
          var_dic = {}
          labels = self.ground_truth.squeeze().long()
          var_dic["CEP"] = nn.CrossEntropyLoss()(self.output, labels)
          return var_dic

        """

    def valid_epoch(self):
        avg_dic = dict()
        self.net.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            self.output = self.net(self.input).detach()
            dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # sum up
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(var_dict=avg_dic, global_step=self.step, tag="Valid")
        self.loger.write(self.step, self.current_epoch, avg_dic, "Valid", header=self.current_epoch <= 1)
        self.net.train()

    def get_data_from_batch(self, batch_data, device):
        """If you have different behavior. You need to rewrite thisd method and the method `sllf.train_epoch()`

        :param batch_data: A Tensor loads from dataset
        :param device: compute device
        :return: Tensors,
        """
        input_tensor, labels_tensor = batch_data[0], batch_data[1]
        return input_tensor, labels_tensor

    def _watch_images(self, tag: str, grid_size: tuple = (3, 3), shuffle=False, save_file=True):
        pass

    @property
    def configure(self):
        config_dic = super(ClassificationTrainer, self).configure
        config_dic["num_class"] = self.num_class
        return config_dic
