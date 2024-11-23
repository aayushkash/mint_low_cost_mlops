import unittest
import torch
from train import train_model
import torch.nn.functional as F

class TestMNISTTraining(unittest.TestCase):
    def test_training_accuracy(self):
        _, accuracy = train_model(epochs=1)
        self.assertGreaterEqual(accuracy, 95.0)

    def test_augmentation_accuracy(self):
        _, accuracy = train_model(epochs=1, use_augmentation=True)
        self.assertGreaterEqual(accuracy, 90.0)

    def test_batch_size_impact(self):
        _, accuracy_small_batch = train_model(epochs=1, batch_size=8)
        _, accuracy_large_batch = train_model(epochs=1, batch_size=32)
        self.assertIsNotNone(accuracy_small_batch)
        self.assertIsNotNone(accuracy_large_batch)

if __name__ == '__main__':
    unittest.main() 