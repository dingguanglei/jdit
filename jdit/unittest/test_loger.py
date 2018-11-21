from unittest import TestCase
import torch
from jdit.optimizer import Optimizer
from jdit.trainer.super import Loger
from jdit.model import Model


class TestLoger(TestCase):
    def test_regist_config(self):
        pass
        log = Loger()
        param = torch.nn.Linear(10, 1)
        opt = Optimizer(param.parameters(), lr=0.999, weight_decay=0.03, momentum=0.5, betas=(0.1, 0.4),
                        opt_name="RMSprop")
        log.regist_config(opt)
        self.assertEqual(log.regist_dict["Optimizer"],
                         {'opt_name': 'RMSprop', 'lr': 0.999, 'momentum': 0.5, 'alpha': 0.99, 'eps': 1e-08,
                          'centered': False, 'weight_decay': 0.03, 'lr_decay': '0.92'})
        opt.do_lr_decay()
        log.regist_config(opt)
        self.assertEqual(log.regist_dict["Optimizer"]["lr"], 0.91908)
        log.regist_config(opt)
        self.assertEqual(log.regist_dict["Optimizer"]["lr"], 0.91908)
        net = Model(torch.nn.Linear(10, 1))
        log.regist_config(net)
        self.assertEqual(log.regist_dict, {
            'Optimizer': {'opt_name': 'RMSprop', 'lr': 0.91908, 'momentum': 0.5, 'alpha': 0.99, 'eps': 1e-08,
                          'centered': False, 'weight_decay': 0.03, 'lr_decay': '0.92'},
            'Model': {'model_name': 'Linear', 'init_method': 'kaiming', 'gpus': 0, 'total_params': 11,
                      'structure': 'Linear(in_features=10, out_features=1, bias=True)'}})
