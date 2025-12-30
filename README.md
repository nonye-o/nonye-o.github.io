---
layout: default
title: "About"
---
# Nonye Okoma
I graduated with an MSci in Mathematics in July 2024 from Lancaster University. Here is my [CV](Gregory_Okoma_CV.pdf).

# Projects
My GitHub repository for these projects is [here](https://github.com/nonye-o/nonye-o.github.io/).

## Python Neural Network from Scratch
My implementation of a highly generalizable Multi-Layer Perceptron (MLP) network in Python from scratch using the NumPy library. Details about the derivation of the gradients and an ablation study on different optimizers (Stochastic Gradient Descent, Momentum, RMSProp, and Adam), different weight initializations (Uniform(-1,1) and He Uniform), and different L2-regularization coefficients can be found [in this report](/NumPy_Neural_Network_Report.pdf). The implementation is contained in the notebook `github NumPy Neural Network From Scratch.ipynb`.

The MNIST handwritten digits CSV dataset can be found [here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv). The Fashion MNIST dataset can be found [here](https://www.kaggle.com/datasets/zalando-research/fashionmnist). The CIFAR-10 CSV dataset can be found [here](https://www.kaggle.com/datasets/fedesoriano/cifar10-python-in-csv). To train a `neural` instance, in the "Engineer Data" section, you need to download one of these datasets and replace `"path_to_data"`
```python
# Load data (you need to download the relevant dataset as a CSV, e.g. CIFAR-10, MNIST digits, or Fashion-MNIST)
data = pd.read_csv("path_to_data", nrows=cutoff*2)
```
with the download path of the CSV file containing the data on your computer. A ready-made `neural` instance named `nn` is available in the "Train a Network" section:
```python
# Select parameters for gradient descent
learning_rate_ = 0.001
minibatch_size_ = 100
epochs_ = 20
optimizer_ = 'adam'
inputs_ = xxx
true_outputs_ = yyy
loss_ = 'categorical cross entropy'
cv_input_layer_ = xxx_cv
cv_true_labels_ = yyy_cv
lambd_ = 0
beta_1_ = 0.9
beta_2_ = 0.999
epsilon_ = 10**(-8)

# Select network architecture
# Note that the activation functions are only for the layers after the first layer since the first layer is the input layer
# Therefore, the activation function list must be one less than the layer dimensions list
layer_dimensions = [xxx.shape[1], 128, 128, 128, 128, yyy.shape[1]]
activation_functions = ['leaky relu', 'leaky relu', 'leaky relu', 'leaky relu', 'softmax']

# Initialize neural network
nn = neural(layers=layer_dimensions, activation_function_list=activation_functions, he_uniform=True)
```
To train the network, just run the cell below in the notebook
```python
# Run gradient descent on the network
nn.gradient_descent(learning_rate_, minibatch_size_, epochs_, optimizer_, inputs_, true_outputs_, loss_, lambd_, cv_input_layer = cv_input_layer_, cv_true_labels = cv_true_labels_, beta_1 = beta_1_, beta_2 = beta_2_, epsilon = epsilon_)
```




## Python Deep Reinforcement Learning
I implemented an Advantage Actor-Critic (A2C) reinforcement learning agent from scratch integrated with an Intrinsic Curiosity Module (ICM) in Python using Gymnasium and PyTorch. This implementation can be found in the file named `RL A2C.py`.
