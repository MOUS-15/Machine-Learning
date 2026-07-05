# Gradient Descent Optimizers

This project implements three of the most popular optimization algorithms used in Machine Learning **from scratch using only NumPy**.

## Implemented Optimizers

- Momentum
- RMSProp
- Adam (with Bias Correction)

## Features

- Mini-Batch Gradient Descent
- Linear Regression from Scratch
- Momentum Optimizer
- RMSProp Optimizer
- Adam Optimizer
- Cost Function Visualization
- Prediction on New Data
- Best Fit Line Visualization

## Mathematical Equations

### Momentum

\[
v = \beta v + (1-\beta)\nabla J
\]

\[
w = w - \eta v
\]

---

### RMSProp

\[
s = \beta s + (1-\beta)(\nabla J)^2
\]

\[
w = w - \eta \frac{\nabla J}{\sqrt{s}+\epsilon}
\]

---

### Adam

First Moment:

\[
m = \beta_1 m + (1-\beta_1)\nabla J
\]

Second Moment:

\[
v = \beta_2 v + (1-\beta_2)(\nabla J)^2
\]

Bias Correction:

\[
\hat{m}=\frac{m}{1-\beta_1^t}
\]

\[
\hat{v}=\frac{v}{1-\beta_2^t}
\]

Parameter Update:

\[
w=w-\eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}
\]

## Requirements

- Python 3.x
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

## Run

```bash
python linear_regression_with_optimizers.py
```

Example Input

```
x:
1,2,3,4,5,6,7,8,9,10

y:
3,5,7,9,11,13,15,17,19,21

Batch Size:
2

Optimizer:
adam
```

Example Output

```
Equation:
y = 1.9985x + 1.0104
```

Predictions

```
x = 2  -> 5.01
x = 4  -> 9.00
x = 20 -> 40.98
```

## Project Structure

```
03-Gradient-Descent-Optimizers
│
├── linear_regression_with_optimizers.py
└── README.md
```

## Author

**Mostafa Fayze**
Faculty of Artificial Intelligence
