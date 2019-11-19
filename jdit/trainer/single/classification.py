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
        """Compute the main loss and observed variables.

        Compute the loss and other caring variables.
        You should return a main loss for doing backward propagation.

        For the caring variables will only be used in tensorboard scalars visualization.
        So, if you want some variables visualized. Make a ``dict()`` with key name is the variable's name.


        .. note::

          Only the main loss will do backward propagation, which is the first returned variable.
          If you have the joint loss, please add them up and return one main loss.

        .. note::

          All of your variables in returned ``dict()`` will never do backward propagation with ``model.train()``.
          However, It still compute grads, without using ``with torch.autograd.no_grad()``.
          So, you can compute any grads variables for visualization.

        Example::

          var_dic = {}
          # visualize the value of CrossEntropyLoss.
          var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

          _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
          total = predict.size(0) * 1.0
          labels = self.labels.squeeze().long()
          correct = predict.eq(labels).cpu().sum().float()
          acc = correct / total
          # visualize the value of accuracy.
          var_dic["ACC"] = acc
          # using CrossEntropyLoss as the main loss for backward, and return by visualized ``dict``
          return loss, var_dic

        """

    @abstractmethod
    def compute_valid(self):
        """Compute the valid_epoch variables for visualization.

        Compute the caring variables.
        For the caring variables will only be used in tensorboard scalars visualization.
        So, if you want some variables visualized. Make a ``dict()`` with key name is the variable's name.

        .. note::

          All of your variables in returned ``dict()`` will never do backward propagation with ``model.eval()``.
          However, It still compute grads, without using ``with torch.autograd.no_grad()``.
          So, you can compute some grads variables for visualization.

        Example::

          var_dic = {}
          # visualize the valid_epoch curve of CrossEntropyLoss
          var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

          _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
          total = predict.size(0) * 1.0
          labels = self.labels.squeeze().long()
          correct = predict.eq(labels).cpu().sum().float()
          acc = correct / total
          # visualize the valid_epoch curve of accuracy
          var_dic["ACC"] = acc
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
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(var_dict=avg_dic, global_step=self.step, tag="Valid")
        self.loger.write(self.step, self.current_epoch, avg_dic, "Valid", header=self.current_epoch <= 1)
        self.net.train()

    def get_data_from_batch(self, batch_data, device):
        input_tensor, labels_tensor = batch_data[0], batch_data[1]
        # if use_onehot:
        #     # label => onehot
        #     y_onehot = torch.zeros(labels.size(0), self.num_class)
        #     if labels.size() != (labels.size(0), self.num_class):
        #         labels = labels.reshape((labels.size(0), 1))
        #     ground_truth_tensor = y_onehot.scatter_(1, labels, 1).long()  # labels =>    [[],[],[]]  batchsize,
        #     num_class
        #     labels_tensor = labels
        # else:
        #     # onehot => label
        #     ground_truth_tensor = labels
        #     labels_tensor = torch.max(self.labels.detach(), 1)

        return input_tensor, labels_tensor

    def _watch_images(self, tag: str, grid_size: tuple = (3, 3), shuffle=False, save_file=True):
        pass

    @property
    def configure(self):
        config_dic = super(ClassificationTrainer, self).configure
        config_dic["num_class"] = self.num_class
        return config_dic
