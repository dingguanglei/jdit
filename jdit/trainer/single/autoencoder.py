from .sup_single import *
from abc import abstractmethod


class AutoEncoderTrainer(SupSingleModelTrainer):
    """this is a autoencoder-decoder trainer. Image to Image

    """

    def __init__(self, logdir, nepochs, gpu_ids, net, opt, datasets):
        super(AutoEncoderTrainer, self).__init__(logdir, nepochs, gpu_ids, net, opt, datasets)
        self.net = net
        self.opt = opt
        self.datasets = datasets

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
            self.input, self.labels = self.get_data_from_batch(batch, self.device)
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
        input_tensor, ground_gruth_tensor = batch_data[0], batch_data[1]
        return input_tensor, ground_gruth_tensor

    def _watch_images(self, tag: str, grid_size: tuple = (3, 3), shuffle=False, save_file=True):
        self.watcher.image(self.input,
                           self.current_epoch,
                           tag="%s/input" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
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
