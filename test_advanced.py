import unittest
import torch
from train import train_model
import torch.nn.functional as F

class TestMNISTTraining(unittest.TestCase):
   
    def test_augmentation_accuracy(self):
        _, accuracy = train_model(epochs=1, use_augmentation=True)
        self.assertGreaterEqual(accuracy, 90.0)

    def test_learning_rates(self):
        # Test different learning rates
        _, accuracy_low_lr = train_model(epochs=1, learning_rate=0.0001)
        _, accuracy_high_lr = train_model(epochs=1, learning_rate=0.01)
        self.assertGreaterEqual(accuracy_low_lr, 85.0, "Low learning rate failed")
        self.assertGreaterEqual(accuracy_high_lr, 85.0, "High learning rate failed")
        
    def test_batch_sizes(self):
        # Test various batch sizes
        batch_sizes = [8, 16, 32]
        for batch_size in batch_sizes:
            _, accuracy = train_model(epochs=1, batch_size=batch_size)
            self.assertGreaterEqual(accuracy, 85.0, 
                                  f"Failed with batch size {batch_size}")

    def test_augmentation_combinations(self):
        # Test different augmentation combinations
        transforms_configs = [
            {"use_augmentation": True, "learning_rate": 0.001, "batch_size": 64},
            {"use_augmentation": True, "learning_rate": 0.005, "batch_size": 32},
            {"use_augmentation": False, "learning_rate": 0.001, "batch_size": 64}
        ]
        
        for config in transforms_configs:
            _, accuracy = train_model(**config)
            self.assertGreaterEqual(accuracy, 85.0, 
                                  f"Failed with config: {config}")

    def test_model_stability(self):
        # Test model stability with different random seeds
        accuracies = []
        for seed in [42, 123, 7]:
            torch.manual_seed(seed)
            _, accuracy = train_model(epochs=1)
            accuracies.append(accuracy)
            
        # Check if accuracy variation is within reasonable bounds
        accuracy_range = max(accuracies) - min(accuracies)
        self.assertLess(accuracy_range, 5.0, 
                       "Model shows high variance across different seeds")

if __name__ == '__main__':
    unittest.main() 