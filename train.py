import torch
import torchvision
import torchvision.transforms as transforms
import os
from model import MNISTNet
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from torchvision.transforms import GaussianBlur

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def create_augmented_samples():
    # Create directories for augmented images
    os.makedirs('augmented_samples', exist_ok=True)
    for i in range(10):
        os.makedirs(f'augmented_samples/{i}', exist_ok=True)

    # Define augmentation transforms with Gaussian blur
    transform = transforms.Compose([
        transforms.RandomRotation(15),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        # transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.ToTensor()
    ])

    # Load MNIST dataset
    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                       download=True, transform=transform)
    
    # Save some augmented samples
    for i in range(10):  # For each digit
        for j in range(5):  # Save 5 samples per digit
            idx = next(idx for idx in range(len(dataset)) 
                      if dataset.targets[idx] == i)
            img, _ = dataset[idx]
            plt.imsave(f'augmented_samples/{i}/sample_{j}.png',
                      img.squeeze().numpy(), cmap='gray')

def train_model(epochs=1, learning_rate=0.001, batch_size=8, use_augmentation=True):
    # Define transforms based on augmentation flag
    if use_augmentation:
        transform = transforms.Compose([
            # transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Added Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MNISTNet()
    
    # Print model parameters
    num_params = count_parameters(model)
    print(f"\nTotal number of parameters: {num_params:,}")
    
    # Change to CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"\nTraining with{'out' if not use_augmentation else ''} augmentation...")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print progress every 100 batches
            if (i + 1) % 100 == 0:
                print(f'Batch [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.3f}, '
                      f'Acc: {100*correct/total:.2f}%', end='\r')
                running_loss = 0.0

        accuracy = 100 * correct / total
        print(f'\nEpoch {epoch + 1}:')
        print(f'Loss: {running_loss/len(train_loader):.3f}')
        print(f'Accuracy: {accuracy:.2f}%\n')
        
    return model, accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--no-augmentation', action='store_true',
                      help='Disable data augmentation')
    parser.add_argument('--create-samples', action='store_true',
                      help='Create augmented samples before training')
    args = parser.parse_args()
    
    if args.create_samples:
        print("Creating augmented samples...")
        create_augmented_samples()
    
    model, accuracy = train_model(use_augmentation=not args.no_augmentation)