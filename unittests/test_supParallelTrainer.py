from unittest import TestCase
from jdit.parallel import SupParallelTrainer
import os
import shutil

class test_print():
    def __init__(self, params):
        self.params = params
    def train(self, *args, **kwargs):
        print(self.params)


class TestSupParallelTrainer(TestCase):

    def tearDown(self):
        logdir = "log_debug/"
        if os._exists(logdir):
            shutil.rmtree(logdir)

    def setUp(self):
        logdir = "log_debug/"
        if os._exists(logdir):
            shutil.rmtree(logdir)

        unfixed_params = [{'task_id': 2, 'depth': 1, 'gpu_ids_abs': []},
                          {'task_id': 1, 'depth': 2, 'gpu_ids_abs': [1, 2]},
                          {'task_id': 1, 'depth': 3, 'gpu_ids_abs': [1, 2]},
                          {'task_id': 2, 'depth': 4, 'gpu_ids_abs': [3, 4]}
                          ]


    def test_train(self):
        unfixed_params = [{'task_id': 2, 'depth': 1, 'gpu_ids_abs': []},
                          {'task_id': 1, 'depth': 2, 'gpu_ids_abs': []}, ]
        pt = SupParallelTrainer(unfixed_params, test_print)
        pt.train()

    def test__add_logdirs_to_unfixed_params(self):
        unfixed_params = [
            {'depth': 1, 'gpu_ids_abs': []},
            {'depth': 2, 'gpu_ids_abs': [1, 2]}
            ]
        final_unfixed_params = [
            {'depth': 1, 'gpu_ids_abs': [], 'logdir': 'plog/depth=1,gpu=[]'},
            {'depth': 2, 'gpu_ids_abs': [1, 2], 'logdir': 'plog/depth=2,gpu=[1, 2]'}
            ]
        pt = SupParallelTrainer([{'task_id': 1, "logdir": "log"}], test_print)
        test_final_unfixed_params_list = pt._add_logdirs_to_unfixed_params(unfixed_params)
        self.assertEqual(final_unfixed_params, test_final_unfixed_params_list)

    def test__convert_to_dirname(self):
        pt = SupParallelTrainer([{'task_id': 1, 'logdir': "log"}], test_print)
        self.assertEqual(pt._convert_to_dirname("abc"), "abc")
        self.assertEqual(pt._convert_to_dirname("123_abc_abc****"), "123_abc_abc")
        self.assertEqual(pt._convert_to_dirname("*<>,/\\:?|abc"), "smallergreater,__%$-abc")
