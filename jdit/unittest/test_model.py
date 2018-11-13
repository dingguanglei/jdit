from unittest import TestCase
from torch.nn import Conv2d
from torch.nn import init,Conv2d,Sequential
from jdit.model import Model
import os
import shutil

class TestModel(TestCase):
    def setUp(self):
        self.mode = Sequential(Conv2d(10, 1, 3, 1, 0))
        self.epoch = 32

    def test_define(self):
        net = Model()
        assert net.model is None
        net.define(self.mode, [], "kaiming", show_structure=False)
        assert net.model is not None

    def test_print_network(self):
        net = Model(self.mode, show_structure=False)
        assert net.model is not None

    def test_weightsInit(self):
        net = Model()
        net.init_fc = init.kaiming_normal_
        self.mode.apply(net._weight_init)

    def test_loadModel(self):
        print(self.mode)
        net = Model(self.mode, show_structure=False)
        net.check_point("tm", self.epoch, "test_model")
        net.load_model("test_model/checkpoint/Model_tm_%d.pth" % self.epoch,
                      "test_model/checkpoint/Weights_tm_%d.pth" % self.epoch)
        dir = "test_model/"
        shutil.rmtree(dir)


    def test_loadPoint(self):
        net = Model(self.mode, show_structure=False)
        net.check_point("tm", self.epoch, "test_model")
        net.load_point("tm", self.epoch, "test_model")
        dir = "test_model/"
        shutil.rmtree(dir)


    def test_checkPoint(self):
        net = Model(self.mode, show_structure=False)
        net.check_point("tm", self.epoch, "test_model")
        dir = "test_model/"
        shutil.rmtree(dir)


    def test_configure(self):
        net = Model(self.mode)
        for k, v in net.configure.items():
            assert k is not None
            assert v is not None
