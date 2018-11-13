from unittest import TestCase
from ..dataset import HandMNIST

class TestHand_mnist(TestCase):
    def test_buildDatasets(self):
        data = HandMNIST("data")
