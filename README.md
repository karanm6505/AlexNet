# AlexNet using PyTorch

## Description

AlexNet is a deep convolutional neural network (CNN) architecture developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. It won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012 and played a crucial role in the advancement of deep learning. This repository provides an implementation of AlexNet using PyTorch in a Jupyter Notebook (`AlexNet.ipynb`), including model training, evaluation, and results.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [References](#references)

## Introduction

AlexNet is a deep convolutional neural network (CNN) architecture that introduced techniques such as ReLU activation, dropout, and overlapping pooling to improve training efficiency and performance. It consists of eight layers: five convolutional layers followed by three fully connected layers.

This repository provides an implementation of AlexNet using PyTorch in a Jupyter Notebook (`AlexNet.ipynb`), including training and evaluation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AlexNet.git
   cd AlexNet
   ```

## Dataset

The implementation uses the ImageNet dataset or a smaller dataset such as CIFAR-10 for demonstration. The dataset will be automatically downloaded using `torchvision.datasets` if not already present.

**Note:** To download the dataset, ensure that you run **Cell Block 3** in the Jupyter Notebook.

## Usage

### Running the Jupyter Notebook

To use the AlexNet model, open the Jupyter Notebook:

```bash
jupyter notebook AlexNet.ipynb
```

Follow the cells in the notebook to train and evaluate the model.

## Training

The notebook contains a training section where you can adjust hyperparameters such as epochs, batch size, and learning rate.

## Evaluation

The trained model can be evaluated using a test dataset within the notebook.

## Results

After training, the model typically achieves a high accuracy on image classification tasks, significantly outperforming previous architectures.

## References

- Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NIPS, 2012.
- [AlexNet Paper](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

---

