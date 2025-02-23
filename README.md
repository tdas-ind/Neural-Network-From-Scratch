# Neural-Network-From-Scratch

This repository contains a minimal implementation of a neural network from scratch, including automatic differentiation (autograd) and backpropagation. The goal of this project is to understand the fundamentals of neural networks, gradient computation, and optimization.

## Overview

The project implements:
- A `Value` class that represents a scalar value and supports basic operations like addition, multiplication, and activation functions (e.g., `tanh`).
- Automatic differentiation using the chain rule to compute gradients.
- A simple neural network with layers and neurons.
- A training loop to optimize the network using gradient descent.

## Key Features

- **Autograd**: The `Value` class automatically computes gradients using the chain rule.
- **Neural Network**: A multi-layer perceptron (MLP) is implemented with customizable layers and neurons.
- **Training**: The network is trained using gradient descent on a small dataset.

## What I Learned

1. **Automatic Differentiation**:
   - How gradients are computed using the chain rule.
   - The role of the `_backward` method in propagating gradients through the computation graph.

2. **Neural Network Basics**:
   - How neurons and layers are constructed.
   - The role of activation functions like `tanh` in introducing non-linearity.

3. **Backpropagation**:
   - How gradients flow backward through the network to update weights and biases in addition, power and multiplication. 
   - The importance of zeroing out gradients before each backward pass.

4. **Optimization**:
   - How gradient descent is used to minimize the loss function.
   - The impact of learning rate on training stability and convergence.

5. **Debugging**:
   - How to debug gradient computation and parameter updates.
   - The importance of checking intermediate values during training.