from unittest import TestCase
from ..dataset import Hand_mnist

class TestHand_mnist(TestCase):
    def test_buildDatasets(self):
        data = Hand_mnist("data")
