from unittest import TestCase
import torch
from torch.nn import init, Conv2d, Sequential
from jdit.model import Model
import shutil


class TestModel(TestCase):
    def setUp(self):
        self.mode = Sequential(Conv2d(10, 1, 3, 1, 0))
        self.epoch = 32

    def test_define(self):
        net = Model(self.mode, [], "kaiming", show_structure=False)
        if torch.cuda.device_count() == 0:
            net = Model(self.mode, [], "kaiming", show_structure=False)
        elif torch.cuda.device_count() == 1:
            net = Model(self.mode, [0], "kaiming", show_structure=False)
        else :
            net = Model(self.mode, [0, 1], "kaiming", show_structure=False)

    def test_save_load_weights(self):
        print(self.mode)
        if torch.cuda.device_count() == 0:
            net = Model(self.mode, [], "kaiming", show_structure=False)
        elif torch.cuda.device_count() == 1:
            net = Model(self.mode, [0], "kaiming", show_structure=False)
        else :
            net = Model(self.mode, [0, 1], "kaiming", show_structure=False)
        net.check_point("tm", self.epoch, "test_model")
        net.load_weights("test_model/checkpoint/Weights_tm_%d.pth" % self.epoch)
        dir = "test_model/"
        shutil.rmtree(dir)

    def test_load_point(self):
        if torch.cuda.device_count() == 0:
            net = Model(self.mode, [], "kaiming", show_structure=False)
        elif torch.cuda.device_count() == 1:
            net = Model(self.mode, [0], "kaiming", show_structure=False)
        else :
            net = Model(self.mode, [0, 1], "kaiming", show_structure=False)

        net.check_point("tm", self.epoch, "test_model")
        net.load_point("tm", self.epoch, "test_model")
        dir = "test_model/"
        shutil.rmtree(dir)

    def test_check_point(self):
        if torch.cuda.is_available():
            net = Model(self.mode, [0], "kaiming", show_structure=False)
        elif torch.cuda.device_count() > 1:
            net = Model(self.mode, [0, 1], "kaiming", show_structure=False)
        elif torch.cuda.device_count() > 2:
            net = Model(self.mode, [2, 3], "kaiming", show_structure=False)
        else:
            net = Model(self.mode, [], "kaiming", show_structure=False)
        net.check_point("tm", self.epoch, "test_model")
        dir = "test_model/"
        shutil.rmtree(dir)

    def test__weight_init(self):
        if torch.cuda.is_available():
            net = Model(self.mode, [0], "kaiming", show_structure=False)
        elif torch.cuda.device_count() > 1:
            net = Model(self.mode, [0, 1], "kaiming", show_structure=False)
        elif torch.cuda.device_count() > 2:
            net = Model(self.mode, [2, 3], "kaiming", show_structure=False)
        else:
            net = Model(self.mode, [], "kaiming", show_structure=False)
        net.init_fc = init.kaiming_normal_
        self.mode.apply(net._weight_init)

    def test__fix_weights(self):
        pass

    def test__set_device(self):
        pass

    def test_configure(self):
        if torch.cuda.device_count() == 0:
            net = Model(self.mode, [], "kaiming", show_structure=False)
        elif torch.cuda.device_count() == 1:
            net = Model(self.mode, [0], "kaiming", show_structure=False)
        else :
            net = Model(self.mode, [0, 1], "kaiming", show_structure=False)
        self.assertEqual(net.configure,
                         {'model_name': 'Sequential', 'init_method': 'kaiming', 'total_params': 91,
                          'structure': 'Sequential(\n  (0): Conv2d(10, 1, kernel_size=(3, 3), '
                                       'stride=(1, 1))\n)'})
        print(net.configure)
        for k, v in net.configure.items():
            assert k is not None
            assert v is not None
