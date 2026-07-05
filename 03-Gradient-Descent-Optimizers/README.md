
This image contains:

- Data points (true values)
- Learned regression line
- Cost function curve over epochs

---

# 📌 Overview

Traditional Gradient Descent often suffers from:

- Slow convergence  
- Oscillations  
- Poor performance on noisy or large datasets  

This project demonstrates how modern optimizers solve these problems by improving:

- Stability  
- Speed  
- Convergence quality  

---

# 📐 Mathematical Formulation

## Linear Model

\[
\hat{y} = wx + b
\]

---

## Cost Function (MSE)

\[
J(w,b) = \frac{1}{2m} \sum (\hat{y} - y)^2
\]

---

## Gradients

\[
dw = \frac{1}{m} \sum (\hat{y} - y)x
\quad,\quad
db = \frac{1}{m} \sum (\hat{y} - y)
\]

---

## Optimizers

### 🔹 Momentum
Smooths updates using velocity:

\[
v = \beta v + (1-\beta)g
\]

---

### 🔹 RMSprop
Adaptive learning rate per parameter:

\[
v = \beta v + (1-\beta)g^2
\]

---

### 🔹 Adam
Combination of Momentum + RMSprop:

- First moment (mean)
- Second moment (variance)
- Bias correction

---

# 📊 Visualization (Core Feature)

The model generates a **2-panel plot**:

### 📍 Left Plot:
- Original data points
- Learned regression line

### 📍 Right Plot:
- Cost function vs Epochs
- Convergence behavior

---

# ✨ Features

- 🔥 Pure NumPy implementation
- ⚙️ 3 Optimizers (Momentum / RMSprop / Adam)
- 📦 Mini-batch Gradient Descent
- ⛔ Early Stopping
- 📊 Full training visualization
- 💻 Interactive CLI input
- 📈 Cost tracking per epoch

---

# 🏗️ Project Structure

- `fit_momentum()` → Momentum training  
- `fit_rmsprop()` → RMSprop training  
- `fit_adam()` → Adam training  
- `predict()` → inference  
- `plot()` → visualization dashboard  
- `run_model()` → full pipeline  

---

# 🚀 Installation & Run

## Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
