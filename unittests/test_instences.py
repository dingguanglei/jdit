from unittest import TestCase
from jdit.trainer.instances import start_fashionClassTrainer, start_fashionGenerateGanTrainer, \
    start_cifarPix2pixGanTrainer, start_fashionAutoencoderTrainer, start_fashionClassPrarallelTrainer
import shutil
import os


class TestInstances(TestCase):
    def test_start_fashionClassTrainer(self):
        start_fashionClassTrainer(run_type="debug")

    def test_start_fashionGenerateGanTrainer(self):
        start_fashionGenerateGanTrainer(run_type="debug")

    def test_start_cifarPix2pixGanTrainer(self):
        start_cifarPix2pixGanTrainer(run_type="debug")

    def test_start_fashionAotoencoderTrainer(self):
        start_fashionAutoencoderTrainer(run_type="debug")

    def test_start_fashionClassPrarallelTrainer(self):
        start_fashionClassPrarallelTrainer()

    def setUp(self):
        dir = "log_debug/"
        if os._exists(dir):
            shutil.rmtree(dir)

    def tearDown(self):
        dir = "log_debug/"
        if os._exists(dir):
            shutil.rmtree(dir)
