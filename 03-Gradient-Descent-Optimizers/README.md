# 🚀 Linear Regression with Advanced Optimizers (From Scratch)

A complete from-scratch implementation of **Linear Regression** using only NumPy, featuring modern optimization algorithms used in deep learning:

- Momentum
- RMSprop
- Adam

All implemented without any machine learning libraries.

---

# 📌 Overview

This project demonstrates how Linear Regression can be trained using different optimization techniques instead of standard Gradient Descent.

It focuses on:

- Understanding optimization mechanics
- Comparing optimizer performance
- Visualizing learning behavior

---

# ⚙️ Features

- Pure NumPy implementation (no ML libraries)
- Mini-batch Gradient Descent
- Momentum optimizer
- RMSprop optimizer
- Adam optimizer with bias correction
- Early stopping for efficient training
- Real-time cost tracking
- Visualization of regression line and loss curve
- Interactive CLI input

---

# 📐 Mathematical Background

## Cost Function (MSE)

\[
J(w,b) = \frac{1}{2m} \sum (\hat{y} - y)^2
\]

\[
\hat{y} = wx + b
\]

---

## Gradients

\[
dw = \frac{1}{m} \sum (\hat{y} - y)x
\]

\[
db = \frac{1}{m} \sum (\hat{y} - y)
\]

---

## Momentum

\[
v = \beta v + (1 - \beta) g
\]

\[
w = w - \eta v
\]

---

## RMSprop

\[
v = \beta v + (1 - \beta) g^2
\]

\[
w = w - \frac{\eta}{\sqrt{v} + \epsilon} g
\]

---

## Adam Optimizer

Combines Momentum + RMSprop with bias correction:

\[
w = w - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \hat{m}
\]

---

# 🏗️ Code Structure

- `__init__` → initialize hyperparameters  
- `fit_momentum()` → training using Momentum  
- `fit_rmsprop()` → training using RMSprop  
- `fit_adam()` → training using Adam  
- `predict()` → make predictions  
- `plot()` → visualization  
- `run_model()` → full pipeline execution  

---

# 🚀 Installation

## 1. Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
