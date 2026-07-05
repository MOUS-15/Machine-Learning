# 🚀 Advanced Linear Regression with Adaptive Optimizers (From Scratch)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-darkgreen?style=for-the-badge&logo=numpy)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange?style=for-the-badge)](https://matplotlib.org/)
[![From Scratch](https://img.shields.io/badge/ML-From_Scratch-red?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A fully from-scratch implementation of **Linear Regression** using only **NumPy**, featuring modern optimization algorithms used in deep learning:

- Momentum  
- RMSprop  
- Adam  

---

# 📊 Visualization (Core Feature)

This project generates a complete **training dashboard** automatically:

### 📌 1. Regression Fit
- Data points (ground truth)
- Learned regression line

### 📌 2. Loss Curve
- Cost vs Epochs
- Convergence behavior

### 🖼 Output Example

The model saves:

Which includes:
- Regression line
- Training data distribution
- Loss curve over time

---

# 📌 Overview

Gradient Descent alone often struggles with:

- Slow convergence
- Oscillations
- Sensitivity to learning rate

This project demonstrates how adaptive optimizers solve these issues by improving:

- Stability
- Speed
- Convergence quality

---

# 📐 Mathematical Framework

## Linear Model
\[
\hat{y} = wx + b
\]

## Cost Function (MSE)
\[
J(w,b) = \frac{1}{2m} \sum (\hat{y} - y)^2
\]

## Gradients
\[
dw = \frac{1}{m} \sum (\hat{y} - y)x
\quad,\quad
db = \frac{1}{m} \sum (\hat{y} - y)
\]

---

## ⚙️ Optimizers

### 🔹 Momentum
\[
v = \beta v + (1 - \beta) g
\]

### 🔹 RMSprop
\[
v = \beta v + (1 - \beta) g^2
\]

### 🔹 Adam
Combines:
- Momentum (1st moment)
- RMSprop (2nd moment)
- Bias correction

---

# ⚖️ Optimizers Comparison

| Optimizer | Strength | Adaptivity | Stability |
|----------|----------|------------|-----------|
| Momentum | Faster convergence | Low | High |
| RMSprop  | Adaptive learning rate | Medium | Very High |
| Adam     | Best overall performance | High | Excellent |

---

# ✨ Key Features

- Pure NumPy implementation (no ML frameworks)
- Mini-batch Gradient Descent
- 3 Optimizers (Momentum / RMSprop / Adam)
- Smart Early Stopping
- Full training pipeline (fit → predict → plot)
- Automatic visualization dashboard
- Cost tracking per epoch
- Clean OOP design

---

# 🏗️ Project Structure

- `fit_momentum()` → Momentum training  
- `fit_rmsprop()` → RMSprop training  
- `fit_adam()` → Adam training  
- `predict()` → inference  
- `plot()` → visualization dashboard  
- `run_model()` → full pipeline  

---

# 🚀 Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install numpy matplotlib
