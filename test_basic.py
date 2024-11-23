import unittest
import torch
from model import MNISTNet

class TestMNISTModel(unittest.TestCase):
    def setUp(self):
        self.model = MNISTNet()

    def test_model_output_shape(self):
        x = torch.randn(1, 1, 28, 28)
        output = self.model(x)
        self.assertEqual(output.shape, (1, 10))

    def test_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertLess(total_params, 25000)

    def test_forward_pass(self):
        x = torch.randn(1, 1, 28, 28)
        output = self.model(x)
        self.assertTrue(torch.isfinite(output).all())

if __name__ == '__main__':
    unittest.main() 