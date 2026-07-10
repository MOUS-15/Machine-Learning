# 🐾 Production-Grade Multi-Class Logistic Regression from Scratch

[![Language](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/NumPy-Pure%20Math-orange.svg)](https://numpy.org/)
[![Visualization](https://img.shields.io/badge/Matplotlib-Dark%20Theme-green.svg)](https://matplotlib.org/)

An exhaustive, industrial-standard documentation and implementation of a **Multinomial Logistic Regression model (Softmax Regression)** built strictly from scratch. This project demonstrates advanced knowledge of vectorized gradient descent, numerical stability practices in machine learning, mathematical data synthesis, and complex 2D decision-boundary evaluation.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Theoretical & Mathematical Foundation](#-theoretical--mathematical-foundation)
3. [Synthetic Data Architecture](#-synthetic-data-architecture)
4. [Software Design & Architecture](#-software-design--architecture)
5. [Numerical Stability & Edge Cases](#-numerical-stability--edge-cases)
6. [Installation & Execution](#-installation--execution)
7. [Comprehensive Output Examples](#-comprehensive-output-examples)
8. [Hyperparameter Tuning Guide](#-hyperparameter-tuning-guide)

---

## 🗺️ Project Overview

In multi-class classification problems, mapping continuous multi-dimensional inputs into discrete categorical probabilities is a foundational challenge. This project implements a self-contained pipeline that:
* Generates a 2-feature synthetic dataset representing three distinct animal classes (**Cats, Dogs, Birds**).
* Applies a vectorized **Softmax Activation Layer**.
* Minimizes **Categorical Cross-Entropy Loss** using batch gradient descent.
* Employs an optimization safety net with a strict mathematical convergence checker (**Early Stopping** via $\epsilon$).
* Renders a complete spatial classification landscape using automated meshgrid boundary mapping.

---

## 📐 Theoretical & Mathematical Foundation

Unlike binary logistic regression which utilizes the Sigmoid function to output a single probability, multi-class classification requires a probability distribution across $C$ target classes.

### 1. Forward Propagation (The Linear Engine)
Given an input matrix $X \in \mathbb{R}^{m \times n}$ (where $m$ is the number of samples and $n$ is the number of features including the bias column), and a weight matrix $W \in \mathbb{R}^{n \times C}$ (where $C$ is the number of classes), the raw logit scores matrix $Z$ is computed via matrix multiplication:

$$Z = X \cdot W$$

### 2. Activation Layer (Softmax Function)
To convert logits into normalized probabilities where $\sum_{j=1}^{C} \hat{y}_{ij} = 1$, we map each element through the Softmax function:

$$\hat{y}_{ic} = \sigma(Z)_{ic} = \frac{e^{z_{ic}}}{\sum_{j=1}^{C} e^{z_{ij}}}$$

### 3. Objective Cost Function (Categorical Cross-Entropy)
To penalize inaccurate predictions, we calculate the average negative log-likelihood across all $m$ samples and $C$ classes:

$$J(W) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$$

Where $y_{ic}$ is a binary indicator ($0$ or $1$) from the One-Hot encoded true label matrix.

### 4. Vectorized Gradient Descent (The Backpropagation)
The partial derivative of the cost function with respect to the weight matrix $W$ yields an incredibly elegant vectorized formulation:

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T \cdot (\hat{y} - y)$$

The weights are then updated iteratively by shifting opposite to the gradient scaled by the learning rate ($\eta$):

$$W := W - \eta \cdot \frac{\partial J}{\partial W}$$

---

## 📊 Synthetic Data Architecture

The script mathematically synthesizes three distinct clusters in a 2D space ($X_1, X_2$) representing custom bounding characteristics. Each cluster is uniformly distributed but strategically separated to simulate clean decision fields:

| Class Index | Label Name | $X_1$ Range (Feature 1) | $X_2$ Range (Feature 2) | Target Matrix Vector (One-Hot) |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Cat** | $[1, 9)$ | $[1, 9)$ | `[1, 0, 0]` |
| **1** | **Dog** | $[7, 16)$ | $[7, 16)$ | `[0, 1, 0]` |
| **2** | **Bird** | $[15, 23)$ | $[15, 23)$ | `[0, 0, 1]` |

*Note: A bias column containing solid $1$s is seamlessly appended as the 3rd feature to allow the hyperplane to shift away from the origin.*

---

## 🏗️ Software Design & Architecture

The system follows a strict **Object-Oriented Programming (OOP)** methodology packaged within a unified structural pipeline:

```text
[run_model]
    │
    ├──> [prepare_data]  --> Synthesizes uniforms, compiles Bias, creates One-Hot encoding.
    ├──> [fit]           --> Loops Epochs -> Softmax -> Weight Updates -> Cost Tracking -> Early Stopping.
    ├──> [accuracy]      --> Computes overall prediction exactness score.
    ├──> [predict]       --> Evaluates raw inputs, calculates exact continuous probabilities.
    ├──> [print_info]    --> Flushes clear status logs directly into the terminal console.
    └──> [draw_data]     --> Generates 2D Meshgrid boundaries & Optimization Decay curves.
