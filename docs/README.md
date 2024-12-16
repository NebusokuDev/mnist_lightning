# MNIST Classification Model - Implementation with PyTorch Lightning

This repository provides a sample implementation for training an image classification model on the MNIST dataset using PyTorch Lightning. It utilizes two different models: `NanoFCNet` (fully connected network) and `NanoConvNet` (convolutional neural network) for the classification task based on the MNIST dataset.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Setup](#setup)
  - [Training the Model](#training-the-model)
  - [Testing the Best Model](#testing-the-best-model)
- [File Structure](#file-structure)
- [License](#license)

## Overview

In this project, the following two models are used to classify MNIST digit images:

- `NanoFCNet`: A simple neural network using only fully connected layers.
- `NanoConvNet`: A more complex neural network structure that includes convolutional layers.

PyTorch Lightning is used to easily manage model training and validation. The best model is saved using the checkpoint feature.

## Dependencies

The dependencies required to run this project are as follows:

- Python 3.7 or higher
- PyTorch
- PyTorch Lightning
- TorchMetrics
- torchvision
- matplotlib (used for plotting graphs if needed)

To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```

The contents of the `requirements.txt` file are as follows:

```txt
torch>=2.0.0
pytorch-lightning>=2.0.0
torchmetrics
torchvision
matplotlib
```

## Usage

### Setup

First, download the required dataset. The MNIST dataset will be automatically downloaded using the `MNISTDataModule` class.

```bash
python train.py
```

### Training the Model

Running `train.py` will train the `NanoConvNet` (convolutional network) model. The training process uses PyTorch Lightning's `Trainer`. During training, accuracy on the validation dataset is monitored, and the best model is saved.

The best model will be saved in the `./checkpoints` directory during the training process.

### Testing the Best Model

After training is complete, you can test the best saved model by running the following code:

```python
# Load the best model
best_model = LitMNIST.load_from_checkpoint(best_model_path, model=NanoFCNet())

# Perform testing
trainer.test(best_model, datamodule=datamodule)
```

## File Structure

```
(...)
  ├── data/                   # Location where the MNIST dataset is stored
  ├── train.py                # Script to train the model
  └── README.md               # This document
```

## License

This project is licensed under the MIT License. Please refer to the `LICENSE` file for details.