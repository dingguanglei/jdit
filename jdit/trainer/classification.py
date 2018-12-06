from torch.nn import CrossEntropyLoss
from .super import *
from abc import abstractmethod
from tqdm import *


class ClassificationTrainer(SupTrainer):
    """this is a classification trainer.

    """
    num_class = None

    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets):
        super(ClassificationTrainer, self).__init__(nepochs, logdir, gpu_ids_abs=gpu_ids)
        self.net = net
        self.opt = opt
        self.datasets = datasets

    def train_epoch(self, subbar_disable=False):
        # self._watch_images(show_imgs_num=3, tag="Train")
        for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1), unit="step", disable=subbar_disable):
            self.step += 1
            self.input, self.ground_truth, self.labels = self.get_data_from_batch(batch, self.device)
            self.output = self.net(self.input)
            self._train_iteration(self.opt, self.compute_loss, csv_filename="Train")

            if iteration == 1:
                self._watch_images(show_imgs_num=3, tag="Train")

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
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return loss, var_dic

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
        var_dic = {}
        # Input: (N,C) where C = number of classes
        # Target: (N) where each value is 0≤targets[i]≤C−1
        # ground_truth = self.ground_truth.long().squeeze()
        var_dic["CEP"] = loss = CrossEntropyLoss()(self.output, self.labels.squeeze().long())

        _, predict = torch.max(self.output.detach(), 1)  # 0100=>1  0010=>2
        total = predict.size(0) * 1.0
        labels = self.labels.squeeze().long()
        correct = predict.eq(labels).cpu().sum().float()
        acc = correct / total
        var_dic["ACC"] = acc
        return var_dic

    def valid_epoch(self):
        avg_dic = {}
        self.net.eval()
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth, self.labels = self.get_data_from_batch(batch, self.device)
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

    def get_data_from_batch(self, batch_data, device, use_onehot=True):
        input_cpu, labels = batch_data[0], batch_data[1]
        if use_onehot:
            # label => onehot
            y_onehot = torch.zeros(labels.size(0), self.num_class)
            if labels.size() != (labels.size(0), self.num_class):
                labels = labels.reshape((labels.size(0), 1))
            ground_truth_cpu = y_onehot.scatter_(1, labels, 1).long()  # labels =>    [[],[],[]]  batchsize,num_class
            labels_cpu = labels
        else:
            # onehot => label
            ground_truth_cpu = labels
            labels_cpu = torch.max(self.labels.detach(), 1)

        return input_cpu.to(device), ground_truth_cpu.to(device), labels_cpu.to(device)

    def _watch_images(self, show_imgs_num=4, tag="Train"):
        pass
    #
    # def _change_lr(self):
    #     self.opt.do_lr_decay()
    #
    # def _check_point(self):
    #     self.net._check_point("classmodel", self.current_epoch, self.logdir)

    # def _record_configs(self):
    #     self.loger.regist_config(self.opt, self.current_epoch)
    #     self.loger.regist_config(self.performance, self.current_epoch)  # for self.performance.configure

    @property
    def configure(self):
        dict = super(ClassificationTrainer, self).configure
        dict["num_class"] = self.num_class
        return dict
