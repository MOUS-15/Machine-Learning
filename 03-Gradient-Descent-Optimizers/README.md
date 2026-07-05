# 🚀 Linear Regression with Advanced Optimizers (From Scratch)

**Python | NumPy | Matplotlib**

An end-to-end implementation of **Linear Regression from scratch** without any ML libraries, with a deep comparison of modern optimizers:

- Momentum  
- RMSprop  
- Adam  

Built on top of Mini-batch Gradient Descent.

---

# 📌 Overview

Traditional Gradient Descent often suffers from:

- Slow convergence  
- Oscillations in loss surface  
- Poor performance on large datasets  

This project demonstrates how modern optimizers improve training stability and speed in a clean, educational, from-scratch implementation.

---

# 📐 Mathematical Formulation

## 1. Cost Function (MSE)

\[
J(w,b) = \frac{1}{2m} \sum (\hat{y}^{(i)} - y^{(i)})^2
\]

\[
\hat{y} = wx + b
\]

---

## 2. Gradients

\[
dw = \frac{1}{m} \sum (\hat{y} - y)x
\]

\[
db = \frac{1}{m} \sum (\hat{y} - y)
\]

---

## 3. Momentum

\[
v = \beta v + (1-\beta) g
\]

\[
w = w - \eta v
\]

---

## 4. RMSprop

\[
v = \beta v + (1-\beta) g^2
\]

\[
w = w - \frac{\eta}{\sqrt{v} + \epsilon} g
\]

---

## 5. Adam Optimizer

Combines Momentum + RMSprop with bias correction:

\[
w = w - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \hat{m}
\]

---

# ✨ Features

- Pure NumPy implementation (no ML libraries)
- Momentum / RMSprop / Adam support
- Mini-batch Gradient Descent
- Smart Early Stopping
- CLI-based interaction
- Automatic visualization (loss + regression line)

---

# 🏗️ Project Structure

- `__init__` → initialize hyperparameters  
- `fit_momentum()` → training with Momentum  
- `fit_rmsprop()` → training with RMSprop  
- `fit_adam()` → training with Adam  
- `predict()` → predictions  
- `plot()` → visualization  
- `run_model()` → pipeline execution  

---

# 🚀 Installation & Usage

## Clone repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
