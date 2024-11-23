import unittest
import torch
from model import MNISTNet
from train import train_model

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

    def test_training_accuracy(self):
        _, accuracy = train_model(epochs=1)
        self.assertGreaterEqual(accuracy, 95.0)

    def test_batch_size_impact(self):
        _, accuracy_small_batch = train_model(epochs=1, batch_size=8)
        _, accuracy_large_batch = train_model(epochs=1, batch_size=32)
        self.assertIsNotNone(accuracy_small_batch)
        self.assertIsNotNone(accuracy_large_batch)

if __name__ == '__main__':
    unittest.main() 