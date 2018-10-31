from unittest import TestCase
from jdit.dataset import Cifar10
from jdit.assessment.fid import FID_score
import os


class TestFID_score(TestCase):
    def test_FID_score(self):
        loader = Cifar10(root=r"../../datasets/cifar10", batch_shape=(128, 3, 32, 32))
        target_tensor = loader.samples_train[0]
        source_tensor = loader.samples_valid[0]
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        fid_value = FID_score(source_tensor, target_tensor, sample_prop=0.1, gpu_ids=[0], dim=768)
        print('FID: ', fid_value)
        fid_value = FID_score(loader.loader_test, loader.loader_valid, gpu_ids=[0], dim=768)
        print('FID: ', fid_value)
