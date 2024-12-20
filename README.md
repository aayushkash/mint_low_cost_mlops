# MNIST Classification with PyTorch

[![Python application](https://github.com/aayushkash/mint_low_cost_mlops/actions/workflows/python-app.yml/badge.svg)](https://github.com/aayushkash/mint_low_cost_mlops/actions/workflows/python-app.yml)

A PyTorch implementation of MNIST digit classification with specific architectural constraints and high accuracy.

## Model Architecture

- Less than 25,000 parameters
- Two convolutional layers (1→16 and 16→25 channels)
- Kernel size of 3x3
- MaxPooling after each ReLU activation
- Two linear layers (1→32 and 32→10)
- Flatten operation and LogSoftmax activation

## Features

- Achieves >95% training accuracy in 1 epoch
- Includes image augmentation (rotation and translation)
- Augmented samples saved in labeled directories
- Comprehensive test suite (basic and advanced)
- Automated testing with GitHub Actions

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib

## Project Structure 