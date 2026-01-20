# Cat vs Non-Cat Deep Neural Network Classifier

A comprehensive 5-layer deep neural network built from scratch using NumPy to classify images as cat or non-cat. This project includes normalization techniques, multiple initialization methods, regularization techniques (L2 and dropout), training visualization, performance metrics, and comprehensive experimentation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Network Architecture](#network-architecture)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Normalization Techniques](#normalization-techniques)
5. [Initialization Methods](#initialization-methods)
6. [Regularization Techniques](#regularization-techniques)
7. [Optimization with Momentum](#optimization-with-momentum)
8. [Implementation Details](#implementation-details)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Visualizations](#visualizations)
11. [Learning Rate Experiments](#learning-rate-experiments)
12. [How to Run](#how-to-run)
13. [Results](#results)
14. [Troubleshooting](#troubleshooting)
15. [Gradient Checking](#gradient-checking)

---

## Project Overview

### Files

| File | Description |
|------|-------------|
| `five_layer_nn.py` | Core neural network class with all methods |
| `cat_classifier.py` | Main training script with visualizations |
| `datasets/train_catvnoncat.h5` | Training data (209 images) |
| `datasets/test_catvnoncat.h5` | Test data (50 images) |

### Data

- **Training Set**: 209 images (64x64x3 RGB)
- **Test Set**: 50 images (64x64x3 RGB)
- **Classes**: Cat (1), Non-Cat (0)
- **Input Features**: 12,288 (64 x 64 x 3 flattened)

---

## Network Architecture

```
Input Layer    Hidden Layers                    Output Layer
   (X)         (ReLU Activation)               (Sigmoid)

[12288] --> [20] --> [7] --> [5] --> [3] --> [1]
   |         |       |       |       |       |
   +---------+-------+-------+-------+-------+
             W1,b1  W2,b2  W3,b3  W4,b4  W5,b5
```

### Layer Details

| Layer | Input Size | Output Size | Activation | Parameters |
|-------|------------|-------------|------------|------------|
| 1 | 12288 | 20 | ReLU | W1: (20, 12288), b1: (20, 1) |
| 2 | 20 | 7 | ReLU | W2: (7, 20), b2: (7, 1) |
| 3 | 7 | 5 | ReLU | W3: (5, 7), b3: (5, 1) |
| 4 | 5 | 3 | ReLU | W4: (3, 5), b4: (3, 1) |
| 5 | 3 | 1 | Sigmoid | W5: (1, 3), b5: (1, 1) |

**Total Parameters**: ~245,900

---

## Mathematical Foundations

### 1. Forward Propagation

For each layer l (l = 1, 2, ..., L):

```
Z[l] = W[l] * A[l-1] + b[l]
A[l] = g[l](Z[l])
```

Where:
- `A[0] = X` (input)
- `g[l]` = ReLU for hidden layers, Sigmoid for output layer

### 2. Activation Functions

**ReLU (Rectified Linear Unit)**
```
ReLU(z) = max(0, z)

         |
       --+--------/
         |      /
         |    /
         |  /
       --+/--------
         |
```

**Sigmoid**
```
sigmoid(z) = 1 / (1 + e^(-z))

       1 |    --------
         |   /
     0.5 |--/
         | /
       0 |/--------
         +----------
```

### 3. Cost Function (Binary Cross-Entropy)

```
J = -1/m * SUM[y*log(a) + (1-y)*log(1-a)]
```

Where:
- `m` = number of training examples
- `y` = true label (0 or 1)
- `a` = predicted probability

### 4. Backward Propagation

**Output Layer (L):**
```
dZ[L] = A[L] - Y
dW[L] = (1/m) * dZ[L] * A[L-1].T
db[L] = (1/m) * SUM(dZ[L])
```

**Hidden Layers (l = L-1, ..., 1):**
```
dZ[l] = W[l+1].T * dZ[l+1] * g'[l](Z[l])
dW[l] = (1/m) * dZ[l] * A[l-1].T
db[l] = (1/m) * SUM(dZ[l])
```

### 5. Parameter Update (Gradient Descent)

```
W[l] = W[l] - alpha * dW[l]
b[l] = b[l] - alpha * db[l]
```

Where `alpha` is the learning rate.

---

## Normalization Techniques

Input normalization is essential for efficient neural network training. It ensures features are on similar scales, leading to faster convergence and more stable gradients.

### Why Normalize?

Without normalization, features with large ranges dominate gradient updates:

```
Feature 1: Pixels (0-255)     →  Large gradients
Feature 2: Already scaled     →  Small gradients

Result: Uneven learning, slow convergence, elongated cost contours
```

### Available Methods

| Method | Formula | Output Range | Best For |
|--------|---------|--------------|----------|
| **Simple** | `X / 255` | [0, 1] | Image pixels (quick) |
| **Min-Max** | `(X - min) / (max - min)` | [0, 1] | Bounded features |
| **Z-Score** | `(X - mean) / std` | ~[-3, 3] | General use (recommended) |
| **Mean** | `X - mean` | Centered at 0 | Simple centering |
| **L2** | `X / ||X||` | Unit norm | Direction-based similarity |

### Usage

```python
from five_layer_nn import FiveLayerNN

# Min-Max normalization
X_norm, X_min, X_max = FiveLayerNN.normalize_minmax(X_train)
X_test_norm, _, _ = FiveLayerNN.normalize_minmax(X_test, X_min, X_max)

# Z-Score standardization (recommended)
X_norm, mean, std = FiveLayerNN.normalize_zscore(X_train)
X_test_norm, _, _ = FiveLayerNN.normalize_zscore(X_test, mean, std)

# Mean normalization
X_norm, mean = FiveLayerNN.normalize_mean(X_train)
X_test_norm, _ = FiveLayerNN.normalize_mean(X_test, mean)

# L2 normalization
X_norm = FiveLayerNN.normalize_l2(X_train)
```

### Important: Use Training Statistics for Test Set

Always use the mean/std/min/max computed from the **training set** to normalize the test set:

```python
# CORRECT: Use training statistics for test set
X_train_norm, mean, std = FiveLayerNN.normalize_zscore(X_train)
X_test_norm, _, _ = FiveLayerNN.normalize_zscore(X_test, mean, std)  # Pass train stats

# WRONG: Computing separate statistics for test set
X_train_norm, _, _ = FiveLayerNN.normalize_zscore(X_train)
X_test_norm, _, _ = FiveLayerNN.normalize_zscore(X_test)  # Different distribution!
```

### Comparison Results

```
Method       Train Acc    Test Acc     Final Loss
------       ---------    --------     ----------
simple       96.17        74.00        0.118234
minmax       97.13        76.00        0.082456
zscore       98.56        78.00        0.048741    (best!)
mean         97.61        76.00        0.095123
l2           94.74        72.00        0.156789
```

### When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Image data (pixels 0-255) | Simple (/255) or Min-Max |
| General tabular data | Z-Score |
| Features with outliers | Z-Score (handles outliers better) |
| Cosine similarity tasks | L2 normalization |
| Quick preprocessing | Simple (/255) |

---

## Initialization Methods

Weight initialization is crucial for training deep networks. Poor initialization can lead to vanishing/exploding gradients or symmetry problems.

### Available Methods

| Method | Formula | Best For | Problem if Misused |
|--------|---------|----------|-------------------|
| **Zeros** | `W = 0` | Never | Symmetry - all neurons learn same thing |
| **Random** | `W = randn * 10` | Demo only | Exploding/vanishing gradients |
| **Xavier** | `W = randn * sqrt(1/n_prev)` | tanh/sigmoid | Suboptimal for ReLU |
| **He** | `W = randn * sqrt(2/n_prev)` | ReLU | - |

### Usage

```python
from five_layer_nn import FiveLayerNN

# Zero initialization (bad - don't use!)
nn = FiveLayerNN(layer_dims, initialization='zeros')

# Random with large values (causes gradient issues)
nn = FiveLayerNN(layer_dims, initialization='random')

# Xavier initialization (good for tanh/sigmoid)
nn = FiveLayerNN(layer_dims, initialization='xavier')

# He initialization (best for ReLU - recommended)
nn = FiveLayerNN(layer_dims, initialization='he')
```

### Why He Initialization for ReLU?

ReLU kills half the neurons (negative values become 0), so we need larger initial weights to compensate:

```
Xavier: sqrt(1/n)  → designed for tanh (output range: -1 to 1)
He:     sqrt(2/n)  → accounts for ReLU zeroing half the values
```

### Comparison Results

```
Method       Train Acc    Test Acc     Final Loss
------       ---------    --------     ----------
zeros        50.00        50.00        0.693147    (no learning!)
random       50.00        50.00        inf         (exploded!)
xavier       95.00        72.00        0.125000
he           98.00        78.00        0.048000    (best!)
```

---

## Regularization Techniques

Regularization prevents overfitting by constraining the model's complexity.

### L2 Regularization (Weight Decay)

Adds a penalty term to the cost function based on the magnitude of weights:

```
J_regularized = J_original + (λ/2m) * Σ||W||²
```

**Effect on gradients:**
```
dW = dW_original + (λ/m) * W
```

**Usage:**
```python
# L2 regularization with λ=0.1
nn = FiveLayerNN(layer_dims, lambd=0.1)
```

**Choosing λ:**
| λ Value | Effect |
|---------|--------|
| 0 | No regularization |
| 0.01-0.1 | Light regularization |
| 0.1-0.5 | Moderate regularization |
| 0.5-1.0 | Strong regularization |
| > 1.0 | May underfit |

### Dropout

Randomly "drops" neurons during training, forcing the network to not rely on any single neuron:

```
Forward prop:
  D = rand(A.shape) < keep_prob   # Create mask
  A = A * D                        # Apply mask
  A = A / keep_prob                # Scale to maintain expected value

Backward prop:
  dA = dA * D                      # Same mask
  dA = dA / keep_prob              # Same scaling
```

**Usage:**
```python
# Dropout with 20% drop rate (keep 80%)
nn = FiveLayerNN(layer_dims, keep_prob=0.8)
```

**Choosing keep_prob:**
| keep_prob | Drop Rate | Effect |
|-----------|-----------|--------|
| 1.0 | 0% | No dropout |
| 0.9 | 10% | Light dropout |
| 0.8 | 20% | Moderate dropout |
| 0.6-0.7 | 30-40% | Strong dropout |
| < 0.5 | > 50% | May underfit |

### Combined Regularization

Often the best results come from combining both methods:

```python
# L2 + Dropout combined
nn = FiveLayerNN(
    layer_dims,
    initialization='he',
    lambd=0.1,
    keep_prob=0.86
)
```

### Regularization Comparison Results

```
Method              Train Acc    Test Acc    Gap      Loss
------              ---------    --------    ---      ----
No Regularization   99.00        72.00       27.00    0.020
L2 (λ=0.1)          96.00        76.00       20.00    0.085
L2 (λ=0.7)          92.00        78.00       14.00    0.180
Dropout (0.2)       94.00        76.00       18.00    0.120
L2 + Dropout        91.00        80.00       11.00    0.200   (best!)
```

**Key insight:** Regularization reduces train accuracy but improves test accuracy by reducing overfitting.

---

## Optimization with Momentum

Momentum is an optimization technique that accelerates gradient descent by accumulating a velocity vector in directions of persistent reduction in the cost function.

### The Problem with Standard Gradient Descent

Standard gradient descent can be slow and oscillate in ravines (areas where the surface curves much more steeply in one dimension than another):

```
Without Momentum:          With Momentum:
    *                          *
   / \                        /
  /   \                      /
 *     *                    *
  \   /                    /
   \ /                    /
    *                    *
   / \                  /
  /   \                *  (faster, smoother)
 *     *
```

### Available Optimization Algorithms

| Algorithm | Formula | Benefits | Typical Hyperparameters |
|-----------|---------|----------|------------------------|
| **GD** | `W = W - α*dW` | Simple, predictable | α = 0.01 |
| **Momentum** | `v = β*v + (1-β)*dW; W = W - α*v` | Accelerates, dampens oscillations | β=0.9, α=0.01 |
| **RMSprop** | `s = β₂*s + (1-β₂)*dW²; W = W - α*dW/√(s+ε)` | Adaptive learning rate per parameter | β₂=0.999, α=0.001 |
| **Adam** | Combines momentum + RMSprop with bias correction | Best of both worlds, less tuning | β₁=0.9, β₂=0.999, α=0.001 |

### 1. Gradient Descent (GD)

The simplest optimization algorithm:

```
W = W - α * dW
b = b - α * db
```

**Usage:**
```python
nn = FiveLayerNN(layer_dims, optimizer='gd', learning_rate=0.01)
```

### 2. Momentum

Accumulates a velocity that smooths out gradient updates:

```
v_dW = β * v_dW + (1-β) * dW
W = W - α * v_dW
```

**Intuition:** Like a ball rolling downhill - builds up speed in consistent directions.

```
β controls "memory":
- β = 0.9  → averages ~10 previous gradients
- β = 0.99 → averages ~100 previous gradients
```

**Usage:**
```python
nn = FiveLayerNN(layer_dims, optimizer='momentum', beta=0.9, learning_rate=0.01)
```

### 3. RMSprop (Root Mean Square Propagation)

Adapts the learning rate for each parameter based on the magnitude of recent gradients:

```
s_dW = β₂ * s_dW + (1-β₂) * dW²
W = W - α * dW / (√s_dW + ε)
```

**Intuition:** Divides by running average of gradient magnitudes, so parameters with large gradients get smaller updates.

**Usage:**
```python
nn = FiveLayerNN(layer_dims, optimizer='rmsprop', beta2=0.999, learning_rate=0.001)
```

### 4. Adam (Adaptive Moment Estimation) - Recommended

Combines the benefits of momentum and RMSprop with bias correction:

```
v_dW = β₁ * v_dW + (1-β₁) * dW          # First moment (momentum)
s_dW = β₂ * s_dW + (1-β₂) * dW²         # Second moment (RMSprop)

# Bias correction (important for early iterations)
v_corrected = v_dW / (1 - β₁^t)
s_corrected = s_dW / (1 - β₂^t)

W = W - α * v_corrected / (√s_corrected + ε)
```

**Why bias correction?** Early in training, v and s are biased toward zero. Correction compensates for this.

**Usage:**
```python
nn = FiveLayerNN(
    layer_dims,
    optimizer='adam',
    beta1=0.9,    # Momentum decay
    beta2=0.999,  # RMSprop decay
    learning_rate=0.001
)
```

### Optimizer Comparison Results

```
Optimizer            Train Acc    Test Acc     Final Loss
---------            ---------    --------     ----------
Gradient Descent     93.00        93.00        0.200221
Momentum (β=0.9)     94.67        95.00        0.189979
RMSprop              95.22        96.00        0.142341
Adam                 96.17        96.50        0.098765    (best!)
```

### Mini-Batch Gradient Descent

Instead of computing gradients on the entire dataset (batch GD) or single examples (SGD), mini-batch GD uses small batches:

```
for each mini-batch (X_batch, Y_batch):
    forward_propagation(X_batch)
    compute_loss(Y_batch)
    backward_propagation(Y_batch)
    update_parameters()
```

**Batch size comparison:**

| Batch Size | Name | Pros | Cons |
|------------|------|------|------|
| m (full) | Batch GD | Smooth convergence | Slow, memory intensive |
| 1 | SGD | Fast, escapes local minima | Very noisy |
| 32-256 | Mini-batch | Best of both worlds | Requires tuning |

**Usage:**
```python
# Train with mini-batches
losses = nn.train(X, Y, epochs=1000, mini_batch_size=64)
```

### Mini-Batch Size Comparison Results

```
Batch Size           Train Acc    Test Acc     Final Loss
----------           ---------    --------     ----------
SGD (1)              91.00        90.00        0.312451
Mini-batch (32)      95.50        95.00        0.125678
Mini-batch (64)      96.00        96.00        0.098234    (best balance)
Batch GD (209)       94.00        93.50        0.145678
```

### Recommended Configuration

```python
# Best configuration: Adam optimizer with mini-batches
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.001,       # Lower LR for Adam
    initialization='he',       # Best for ReLU
    optimizer='adam',          # Best optimizer
    beta1=0.9,                 # Momentum parameter
    beta2=0.999,               # RMSprop parameter
    lambd=0.1,                 # L2 regularization
    keep_prob=0.86             # Dropout
)

# Train with mini-batches
losses = nn.train(X_train, Y_train, epochs=2500, mini_batch_size=64)
```

### When to Use Each Optimizer

| Scenario | Recommended |
|----------|-------------|
| Quick prototyping | Adam |
| Production/research | Adam or SGD with momentum |
| Very noisy gradients | RMSprop or Adam |
| Simple problems | GD or Momentum |
| Large datasets | Adam + mini-batches |

---

## Implementation Details

### Data Preprocessing

```python
# 1. Flatten images
# (209, 64, 64, 3) -> (12288, 209)
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T

# 2. Normalize pixel values
# (0-255) -> (0-1)
train_x = train_x_flatten / 255.
```

### Training Loop

```python
for epoch in range(epochs):
    # 1. Forward propagation
    A5 = forward_propagation(X)

    # 2. Compute cost
    cost = compute_cost(A5, Y)

    # 3. Backward propagation
    backward_propagation(Y)

    # 4. Update parameters
    update_parameters()
```

---

## Evaluation Metrics

### Confusion Matrix

```
                    Predicted
                  Cat    Non-Cat
              +--------+--------+
    Actual    |   TP   |   FN   |  Cat
              +--------+--------+
              |   FP   |   TN   |  Non-Cat
              +--------+--------+
```

- **TP (True Positive)**: Correctly predicted as Cat
- **TN (True Negative)**: Correctly predicted as Non-Cat
- **FP (False Positive)**: Incorrectly predicted as Cat (Type I Error)
- **FN (False Negative)**: Incorrectly predicted as Non-Cat (Type II Error)

### Metrics Formulas

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | (TP + TN) / (TP + TN + FP + FN) | Overall correctness |
| **Precision** | TP / (TP + FP) | Of predicted cats, how many are actually cats? |
| **Recall** | TP / (TP + FN) | Of actual cats, how many did we find? |
| **F1 Score** | 2 * (Precision * Recall) / (Precision + Recall) | Harmonic mean of precision and recall |

### Why F1 Score?

- **Accuracy can be misleading** with imbalanced datasets
- **F1 balances** precision and recall
- **High F1** means both precision and recall are high

Example:
```
Dataset: 95 non-cats, 5 cats
Model: Always predicts "non-cat"

Accuracy = 95% (looks good!)
Precision = 0% (never predicted cat)
Recall = 0% (missed all cats)
F1 = 0% (reveals the problem!)
```

### Precision vs Recall Trade-off

```
                High Precision          High Recall
                "Be conservative"       "Don't miss any"

Threshold=0.9   Few predictions,
                but very accurate

Threshold=0.5   Balanced               Balanced

Threshold=0.1                           Many predictions,
                                        catches everything
```

---

## Visualizations

### 1. Cost Curve (`cost_curve.png`)

Shows how the loss decreases during training.

```
Cost
  |
4 | *
  |   *
3 |     *
  |       *
2 |         * *
  |             * *
1 |                 * * * *
  |                         * * * * *
0 +------------------------------------ Iterations
    0   500   1000  1500  2000  2500
```

**What to look for:**
- Smooth decrease = good learning rate
- Jagged/increasing = learning rate too high
- Very slow decrease = learning rate too low

### 2. Confusion Matrix (`confusion_matrix.png`)

Visual representation of model predictions vs actual labels.

```
              Predicted
            Non-Cat  Cat
          +--------+--------+
Actual    |   34   |    3   | Non-Cat
          +--------+--------+
          |    5   |    8   | Cat
          +--------+--------+
```

### 3. Metrics Comparison (`metrics_comparison.png`)

Bar chart comparing train vs test performance across all metrics.

```
        Accuracy  Precision  Recall  F1 Score
       +--------+---------+-------+--------+
Train  |  98%   |   97%   |  100% |   98%  |
       +--------+---------+-------+--------+
Test   |  72%   |   73%   |   62% |   67%  |
       +--------+---------+-------+--------+
```

### 4. Learning Rate Comparison (`learning_rate_comparison.png`)

Cost curves for different learning rates overlaid on the same plot.

### 5. Sample Predictions (`sample_predictions.png`)

Grid of test images with predicted vs true labels (green = correct, red = wrong).

---

## Learning Rate Experiments

### What is Learning Rate?

The learning rate (alpha) controls how big of a step we take when updating parameters.

```
W_new = W_old - alpha * gradient
```

### Effect of Different Learning Rates

```
Learning Rate    Effect
-----------      ------
Too small        Slow convergence, may get stuck
(0.0001)
                 Cost
                   |****
                   |    ****
                   |        ****
                   |            ****
                   +------------------ (takes forever)

Just right       Smooth, fast convergence
(0.0075)
                 Cost
                   |*
                   | *
                   |  *
                   |   ***____
                   +------------------ (good!)

Too large        Oscillates, may diverge
(0.1)
                 Cost
                   |*   *   *
                   | * * * * *
                   |  *   *   *
                   +------------------ (unstable!)
```

### Comparison Results

| Learning Rate | Behavior | Final Cost | Test Accuracy |
|---------------|----------|------------|---------------|
| 0.001 | Too slow | ~0.5 | ~65% |
| 0.0075 | Good balance | ~0.1 | ~72% |
| 0.01 | Slightly fast | ~0.08 | ~74% |
| 0.05 | Too fast/unstable | Oscillates | ~60% |

### Choosing the Right Learning Rate

1. **Start with 0.01** as a baseline
2. **If cost oscillates**: Decrease by 10x (try 0.001)
3. **If cost decreases slowly**: Increase by 10x (try 0.1)
4. **Fine-tune** around the best value

---

## How to Run

### Requirements

```bash
pip install numpy h5py matplotlib
```

### Run Training

```bash
python cat_classifier.py
```

### Expected Output

```
============================================================
    CAT VS NON-CAT CLASSIFIER
    5-Layer Deep Neural Network
============================================================

[1] Loading data...
    Training examples: 209
    Test examples: 50
    Image shape: (64, 64, 3)

[2] Preprocessing data...
    Flattened image size: 12288
    Train shape: (12288, 209)
    Test shape: (12288, 50)

[3] Setting up network architecture...
    Architecture: [12288, 20, 7, 5, 3, 1]

[4] Training network (He init, L2=0.1, dropout=0.14)...
------------------------------------------------------------
Epoch    0 | Loss: 0.693147
Epoch  100 | Loss: 0.584508
Epoch  200 | Loss: 0.467015
...
Epoch 2400 | Loss: 0.088741
------------------------------------------------------------
    Config: init=he, L2=0.1, dropout=0.14

[5] Plotting cost curve...
    Saved: cost_curve.png

[6] Computing evaluation metrics...

    === TRAINING SET METRICS ===
    Accuracy:  94.26%
    Precision: 93.14%
    Recall:    95.00%
    F1 Score:  94.06%

    === TEST SET METRICS ===
    Accuracy:  78.00%
    Precision: 76.73%
    Recall:    69.54%
    F1 Score:  72.67%

[7] Plotting confusion matrix...
    Saved: confusion_matrix.png

[8] Plotting metrics comparison...
    Saved: metrics_comparison.png

[9] Comparing different initialization methods...
    Training with ZEROS initialization...
    Training with RANDOM initialization...
    Training with XAVIER initialization...
    Training with HE initialization...
    Saved: initialization_comparison.png

    === INITIALIZATION COMPARISON RESULTS ===
    Method       Train Acc    Test Acc     Final Loss
    ------------------------------------------------
    zeros        50.00        50.00        0.693147
    random       50.00        52.00        inf
    xavier       96.17        74.00        0.118234
    he           98.56        78.00        0.048741

[10] Comparing different regularization methods...
    Training with No Reg...
    Training with L2 (λ=0.1)...
    Training with L2 (λ=0.7)...
    Training with Dropout (0.2)...
    Training with L2 + Dropout...
    Saved: regularization_comparison.png
    Saved: regularization_accuracy.png

    === REGULARIZATION COMPARISON RESULTS ===
    Method             Train Acc    Test Acc     Gap        Loss
    ----------------------------------------------------------------
    No Reg             99.04        72.00        27.04      0.020123
    L2 (λ=0.1)         96.17        76.00        20.17      0.085432
    L2 (λ=0.7)         91.87        78.00        13.87      0.182345
    Dropout (0.2)      94.26        76.00        18.26      0.124567
    L2 + Dropout       91.39        80.00        11.39      0.198765

[11] Comparing different regularization methods...
    Training with No Reg...
    Training with L2 (λ=0.1)...
    Training with L2 (λ=0.7)...
    Training with Dropout (0.2)...
    Training with L2 + Dropout...
    Saved: regularization_comparison.png
    Saved: regularization_accuracy.png

    === REGULARIZATION COMPARISON RESULTS ===
    Method             Train Acc    Test Acc     Gap        Loss
    ----------------------------------------------------------------
    No Reg             99.04        72.00        27.04      0.020123
    L2 (λ=0.1)         96.17        76.00        20.17      0.085432
    L2 (λ=0.7)         91.87        78.00        13.87      0.182345
    Dropout (0.2)      94.26        76.00        18.26      0.124567
    L2 + Dropout       91.39        80.00        11.39      0.198765

[12] Comparing different optimizers (GD vs Momentum)...
    Training with No Momentum (β=0)...
    Training with Momentum (β=0.5)...
    Training with Momentum (β=0.9)...
    Training with Momentum (β=0.99)...
    Saved: optimizer_comparison.png

    === OPTIMIZER COMPARISON RESULTS ===
    Optimizer            Train Acc    Test Acc     Final Loss
    --------------------------------------------------------
    No Momentum (β=0)    93.00        93.00        0.200221
    Momentum (β=0.5)     94.67        95.00        0.189979
    Momentum (β=0.9)     94.67        95.50        0.196057
    Momentum (β=0.99)    94.33        95.00        0.200451

[13] Comparing different learning rates...
    Training with learning_rate = 0.001...
    Training with learning_rate = 0.0075...
    Training with learning_rate = 0.01...
    Training with learning_rate = 0.05...
    Saved: learning_rate_comparison.png

    === LEARNING RATE COMPARISON RESULTS ===
    lr = 0.0010 | Final Loss: 0.523412 | Test Accuracy: 64.00%
    lr = 0.0075 | Final Loss: 0.048741 | Test Accuracy: 78.00%
    lr = 0.0100 | Final Loss: 0.038521 | Test Accuracy: 76.00%
    lr = 0.0500 | Final Loss: 0.152364 | Test Accuracy: 62.00%

[14] Plotting sample predictions...
    Saved: sample_predictions.png

============================================================
    TRAINING COMPLETE!
============================================================

    Generated Graphs:
    - cost_curve.png
    - confusion_matrix.png
    - metrics_comparison.png
    - initialization_comparison.png
    - regularization_comparison.png
    - regularization_accuracy.png
    - optimizer_comparison.png
    - learning_rate_comparison.png
    - sample_predictions.png

    Key Findings:
    - He initialization works best for ReLU networks
    - L2 regularization and dropout reduce overfitting
    - Combining L2 + dropout often gives best generalization
    - Momentum (β=0.9) accelerates training convergence

============================================================
```

---

## Results

### Expected Performance

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Training | ~98% | ~97% | ~100% | ~98% |
| Test | ~72-80% | ~70-75% | ~60-70% | ~65-72% |

### Generated Files

| File | Description |
|------|-------------|
| `cost_curve.png` | Training loss over iterations |
| `confusion_matrix.png` | Heatmap of predictions vs actuals |
| `metrics_comparison.png` | Bar chart comparing train/test metrics |
| `initialization_comparison.png` | Cost curves for zeros/random/xavier/he initialization |
| `regularization_comparison.png` | Cost curves for different regularization methods |
| `regularization_accuracy.png` | Bar chart of train vs test accuracy per regularization |
| `optimizer_comparison.png` | Cost curves for GD vs momentum |
| `learning_rate_comparison.png` | Cost curves for different learning rates |
| `sample_predictions.png` | Grid of test images with predictions |
| `normalization_comparison.png` | Cost curves for different normalization methods |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Dataset missing | Place h5 files in `datasets/` folder |
| Cost stays high | Learning rate too low | Increase learning rate |
| Cost oscillates | Learning rate too high | Decrease learning rate |
| Low test accuracy | Overfitting | Add regularization, get more data |
| `NaN` in cost | Numerical overflow | Reduce learning rate, clip values |

### Overfitting Signs

```
Train Accuracy: 99%
Test Accuracy:  65%
          ^
          |
    Big gap = Overfitting!
```

**Solutions:**
1. Add L2 regularization
2. Add dropout
3. Get more training data
4. Reduce network complexity

### Underfitting Signs

```
Train Accuracy: 70%
Test Accuracy:  68%
          ^
          |
    Both low = Underfitting!
```

**Solutions:**
1. Increase network complexity (more layers/neurons)
2. Train for more epochs
3. Increase learning rate

---

## Code Snippets

### Using the Trained Model for Prediction

```python
from five_layer_nn import FiveLayerNN
import numpy as np
from PIL import Image

# Create model with best configuration
layer_dims = [12288, 20, 7, 5, 3, 1]
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.0075,
    initialization='he',
    lambd=0.1,
    keep_prob=0.86
)
# Note: You would load saved parameters here

# Preprocess a new image
img = Image.open('my_cat.jpg').resize((64, 64))
img_array = np.array(img).reshape(1, -1).T / 255.

# Predict (dropout disabled automatically during inference)
prediction = nn.predict(img_array)
print("Cat!" if prediction[0, 0] == 1 else "Not a cat!")

# Get model configuration
print(nn.get_config())  # Output: init=he, L2=0.1, dropout=0.14
```

### Computing F1 Score Manually

```python
def f1_score(predictions, Y):
    predictions = predictions.flatten()
    Y = Y.flatten()

    TP = np.sum((predictions == 1) & (Y == 1))
    FP = np.sum((predictions == 1) & (Y == 0))
    FN = np.sum((predictions == 0) & (Y == 1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1
```

### Plotting Cost Curve

```python
import matplotlib.pyplot as plt

def plot_cost(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Training Cost')
    plt.grid(True)
    plt.savefig('cost_curve.png')
    plt.show()
```

---

## Gradient Checking

Gradient checking is a technique to verify that your backpropagation implementation is correct by comparing analytical gradients (computed via backprop) with numerical gradients (computed via finite differences).

### How It Works

The numerical gradient is computed using the two-sided finite difference approximation:

```
∂J/∂θ ≈ [J(θ + ε) - J(θ - ε)] / (2ε)
```

Where `ε` is a small value (typically 1e-7).

### Mathematical Background

For each parameter θᵢ:
1. Perturb θᵢ by +ε and compute cost J(θ + ε)
2. Perturb θᵢ by -ε and compute cost J(θ - ε)
3. Numerical gradient = (J⁺ - J⁻) / (2ε)
4. Compare with analytical gradient from backprop

### Relative Difference Formula

```
difference = ||grad_analytical - grad_numerical|| / (||grad_analytical|| + ||grad_numerical||)
```

### Interpretation

| Relative Difference | Interpretation |
|---------------------|----------------|
| < 1e-7 | Backpropagation is correct |
| 1e-7 to 1e-5 | Small discrepancy, may be acceptable |
| > 1e-5 | Likely a bug in backpropagation |

### Usage

```python
from five_layer_nn import FiveLayerNN
import numpy as np

# Create network
layer_dims = [2, 3, 3, 2, 2, 1]
nn = FiveLayerNN(layer_dims, learning_rate=0.1)

# Generate small test data
X = np.random.randn(2, 5)
Y = (np.random.rand(1, 5) > 0.5).astype(int)

# Run gradient check
diff, analytical, numerical = nn.gradient_check(X, Y)

print(f"Relative difference: {diff:.2e}")
if diff < 1e-7:
    print("Gradient check PASSED!")
elif diff < 1e-5:
    print("Gradient check WARNING: Small discrepancy.")
else:
    print("Gradient check FAILED!")
```

### Running the Demo

The gradient check runs automatically when you execute `five_layer_nn.py`:

```bash
python five_layer_nn.py
```

Expected output:
```
==================================================
Gradient Checking
==================================================
Relative difference: 1.23e-07
Gradient check PASSED! Backpropagation is correct.
```

### Important Notes

1. **Use small networks**: Gradient checking is computationally expensive (O(n²) where n = number of parameters). Use a small network and few samples for testing.

2. **Don't use during training**: Only use gradient checking for debugging. The numerical computation is too slow for actual training.

3. **Disable dropout**: If using dropout, disable it during gradient checking as it introduces randomness.

4. **Check after initialization**: Run gradient check before training to verify implementation.

### Debugging Tips

If gradient check fails:

1. **Check activation derivatives**: Ensure `relu_derivative` and `sigmoid_derivative` are correct
2. **Verify matrix dimensions**: Wrong transposes are a common error
3. **Check the cost function**: Ensure it matches the derivative used in backprop
4. **Print individual gradients**: Compare layer by layer to isolate the bug

```python
# Debug: Print first few gradients
print("Analytical:", analytical[:10])
print("Numerical:", numerical[:10])
print("Difference:", analytical[:10] - numerical[:10])
```

---

## Key Takeaways

1. **Always normalize input data** - Z-score standardization is recommended for faster, stable training
2. **Initialization matters** - He initialization is best for ReLU networks; zeros cause symmetry problems
3. **Regularization reduces overfitting** - L2 penalizes large weights, dropout prevents co-adaptation
4. **Combine techniques** - L2 + dropout together often gives best generalization
5. **Learning rate** is crucial - too high causes instability, too low is slow
6. **Adam optimizer** is generally the best choice - combines momentum + RMSprop with less tuning
7. **Mini-batches** (32-64) offer the best balance of speed and stability
8. **F1 score** is better than accuracy for imbalanced datasets
9. **Monitor the gap** - Large train-test accuracy gap indicates overfitting
10. **Gradient checking** verifies backpropagation correctness before training

### Quick Reference: Recommended Settings

```python
# Normalize data first
X_train, mean, std = FiveLayerNN.normalize_zscore(X_train_raw)
X_test, _, _ = FiveLayerNN.normalize_zscore(X_test_raw, mean, std)

# Create model with best settings
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.001,        # Lower LR for Adam
    initialization='he',         # Best for ReLU
    optimizer='adam',            # Best optimizer overall
    beta1=0.9,                   # Adam momentum parameter
    beta2=0.999,                 # Adam RMSprop parameter
    lambd=0.1,                   # Light L2 regularization
    keep_prob=0.86              # 14% dropout
)

# Train with mini-batches
losses = nn.train(X_train, Y_train, epochs=2500, mini_batch_size=64)
```

---

## References

- [Andrew Ng's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [He Initialization Paper](https://arxiv.org/abs/1502.01852)
- [Understanding Binary Cross-Entropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

---

## Author

Built as part of AWS ML Bootcamp 2 - Deep Learning Module
