from unittest import TestCase
from jdit.trainer.instances import start_fashingClassTrainer, start_fashingGenerateGanTrainer, \
    start_cifarPix2pixGanTrainer
import shutil
import os


class TestInstances(TestCase):
    @staticmethod
    def test_start_fashingClassTrainer():
        start_fashingClassTrainer(run_type="debug")

    @staticmethod
    def test_start_fashingGenerateGanTrainer():
        start_fashingGenerateGanTrainer(run_type="debug")

    @staticmethod
    def test_start_cifarPix2pixGanTrainer():
        start_cifarPix2pixGanTrainer(run_type="debug")

    def setUp(self):
        dir = "log_debug/"
        if os._exists(dir):
            shutil.rmtree(dir)

    def tearDown(self):
        dir = "log_debug/"
        if os._exists(dir):
            shutil.rmtree(dir)
