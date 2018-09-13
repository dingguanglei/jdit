# coding=utf-8
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
class Loss(object):
    __metaclass__ = ABCMeta
    def __init__(self, watch_dic):
        self.loss  = self.loss
        self.var_dic = watch_dic
        self.result = OrderedDict()

    @abstractmethod
    def loss(self, input, output, ground_truth, is_train=True):
        pass

    def comput_loss(self, input, output, ground_truth):
        self.loss(input, output, ground_truth)

    def gp(self, input, output, ground_truth, is_train=True, model=None):
        gp = None
        return gp


def test():
    loss = Loss()
