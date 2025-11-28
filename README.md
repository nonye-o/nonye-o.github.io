---
title: "Projects"
permalink: "/projects/"
layout: page
---
## Python Neural Network from Scratch
My implementation of a highly generalizable Multi-Layer Perceptron (MLP) network in Python from scratch using the NumPy library. Details about the derivation of the gradients and an ablation study on different optimizers (Stochastic Gradient Descent, Momentum, RMSProp, and Adam), different weight initializations (Uniform(-1,1) and He Uniform), and different L2-regularization coefficients can be found [in this report](/NumPy_Neural_Network_Report.pdf). The implementation is contained in the notebook "github NumPy Neural Network From Scratch.ipynb".

The MNIST handwritten digits CSV dataset can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). The Fashion MNIST dataset can be found [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist). The CIFAR-10 CSV dataset can be found [here](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv). To train a "neural" instance, you need to download one of these datasets and replace "path_to_data"
```
data = pd.read_csv("path_to_data", nrows=cutoff*2)
```
with the download path of the CSV file containing the data on your computer.

## Deep Reinforcement Learning
I implemented an Advantage Actor-Critic (A2C) reinforcement learning agent from scratch integrated with an Intrinsic Curiosity Module (ICM) in Python using Gymnasium and PyTorch. Thi
