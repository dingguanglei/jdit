from unittest import TestCase
from ..parallel import SupParallelTrainer

class TestSupParallelTrainer(TestCase):
    def setUp(self):
        self.default_params = {'data_root': r"datasets/fashion_data",
                               'gpu_ids_abs': [],
                               'depth': 4,
                               'logdir': r"log/tresnet_24d_16m_1"}
        self.unfixed_params = [{'depth': 1, 'gpu_ids_abs': []},
                               {'depth': 2, 'gpu_ids_abs': [1, 2]},
                               {'depth': 3, 'gpu_ids_abs': [1, 2]},
                               {'depth': 4, 'gpu_ids_abs': [3, 4]}
                               ]
        self.final_unfixed_params = [
            {'depth': 1, 'gpu_ids_abs': [],
             'logdir': 'plog/depth=1_mid_channels=1_gpu_ids_abs=[]'},
            {'depth': 2, 'gpu_ids_abs': [1, 2],
             'logdir': 'plog/depth=2_mid_channels=2_gpu_ids_abs=[1, 2]'
             },
            {'depth': 3, 'gpu_ids_abs': [1, 2],
             'logdir': 'plog/depth=3_mid_channels=3_gpu_ids_abs=[1, 2]'
             },
            {'depth': 4, 'gpu_ids_abs': [3, 4],
             'logdir': 'plog/depth=4_mid_channels=4_gpu_ids_abs=[3, 4]'
             }
            ]
        self.candidate_params_list = [
            {'data_root': r"datasets/fashion_data",
             'gpu_ids_abs': [],
             'depth': 1,
             'logdir': r"log/tresnet_24d_16m_1"
             },
            {'data_root': r"datasets/fashion_data",
             'gpu_ids_abs': [1, 2],
             'depth': 2,
             'logdir': r"log/tresnet_24d_16m_1"
             },
            {'data_root': r"datasets/fashion_data",
             'gpu_ids_abs': [1, 2],
             'depth': 3,
             'logdir': r"log/tresnet_24d_16m_1"
             },
            {'data_root': r"datasets/fashion_data",
             'gpu_ids_abs': [3, 4],
             'depth': 4,
             'logdir': r"log/tresnet_24d_16m_1"
             }
            ]
        self.candidate_gpu_ids_abs_list = [[], [1, 2], [1, 2], [3, 4]]
        self.title_list = ["task_A", "task_B", "task_C", "task_D"]
        self.default_title_list = [
            "depth=1_gpu_ids_abs=[]",
            "depth=2_gpu_ids_abs=[1, 2]",
            "depth=3_gpu_ids_abs=[1, 2]",
            "depth=4_gpu_ids_abs=[3, 4]"]
        self.trainers_list = ["Trainer_A", "Trainer_B", "Trainer_C", "Trainer_D"]
        self.pt = SupParallelTrainer(self.default_params, self.unfixed_params)

    def test_build_task_trainer(self):
        self.fail()

    def test_train(self):
        self.fail()

    def test__start_train(self):
        self.fail()

    def test__distribute_task_on_devices(self):
        pass
        candidate_params_list = [
            {'gpu_ids_abs': [],
             'depth': 1,
             'logdir': r"log/tresnet_24d_16m_1"
             },
            {'gpu_ids_abs': [1, 2],
             'depth': 2,
             'logdir': r"log/tresnet_24d_16m_1"
             },
            {'gpu_ids_abs': [1, 2],
             'depth': 3,
             'logdir': r"log/tresnet_24d_16m_1"
             },
            {'gpu_ids_abs': [3, 4],
             'depth': 4,
             'logdir': r"log/tresnet_24d_16m_1"
             }
            ]
        real_gpu_used_plan = {
            (): [{'gpu_ids_abs': [],
                  'depth': 1,
                  'logdir': r"log/tresnet_24d_16m_1"}],
            (1, 2): [{'gpu_ids_abs': [1, 2],
                      'depth': 2,
                      'logdir': r"log/tresnet_24d_16m_1"
                      },
                     {'gpu_ids_abs': [1, 2],
                      'depth': 3,
                      'logdir': r"log/tresnet_24d_16m_1"
                      }],
            (3, 4): [{'gpu_ids_abs': [3, 4],
                      'depth': 4,
                      'logdir': r"log/tresnet_24d_16m_1"
                      }],
            }
        gpu_used_plan = self.pt._distribute_task_on_devices(candidate_params_list)
        self.assertEqual(real_gpu_used_plan, gpu_used_plan)

    def test__build_candidate_params(self):
        default_params = {'gpu_ids_abs': [],
                          'depth': 4,
                          'logdir': r"log/tresnet_24d_16m_1"}
        unfixed_params = [{'depth': 1, 'gpu_ids_abs': []},
                          {'depth': 2, 'gpu_ids_abs': [1, 2]},
                          {'depth': 3, 'gpu_ids_abs': [1, 2]},
                          {'depth': 4, 'gpu_ids_abs': [3, 4]}
                          ]
        candidate_params = [{'gpu_ids_abs': [], 'depth': 1, 'logdir': 'plog/depth=1_gpu_ids_abs=[]'},
                            {'gpu_ids_abs': [1, 2], 'depth': 2, 'logdir': 'plog/depth=2_gpu_ids_abs=[1, 2]'},
                            {'gpu_ids_abs': [1, 2], 'depth': 3, 'logdir': 'plog/depth=3_gpu_ids_abs=[1, 2]'},
                            {'gpu_ids_abs': [3, 4], 'depth': 4, 'logdir': 'plog/depth=4_gpu_ids_abs=[3, 4]'}]
        total_params = self.pt._build_candidate_params(default_params, unfixed_params)
        self.assertEqual(candidate_params, total_params, "not equal!")

    def test__add_logdirs_to_unfixed_params(self):
        unfixed_params = [
            {'depth': 1, 'gpu_ids_abs': []},
            {'depth': 2, 'gpu_ids_abs': [1, 2]}
            ]
        final_unfixed_params = [
            {'depth': 1, 'gpu_ids_abs': [], 'logdir': 'plog/depth=1_gpu_ids_abs=[]'},
            {'depth': 2, 'gpu_ids_abs': [1, 2], 'logdir': 'plog/depth=2_gpu_ids_abs=[1, 2]'}
            ]
        test_final_unfixed_params_list = self.pt._add_logdirs_to_unfixed_params(unfixed_params)
        self.assertEqual(final_unfixed_params, test_final_unfixed_params_list)

    def test__convert_to_dirname(self):
        self.assertEqual(self.pt._convert_to_dirname("abc"), "abc")
        self.assertEqual(self.pt._convert_to_dirname("123_abc_abc****"), "123_abc_abc")
        self.assertEqual(self.pt._convert_to_dirname("*<>,/\\:?|abc"), "smallergreater___%$-abc")

    def test_finish(self):
        self.fail()

    def test_error(self):
        self.fail()

