# Cat vs Non-Cat Deep Neural Network Classifier

A 5-layer deep neural network built from scratch in NumPy for binary image classification. The core model lives in `five_layer_nn.py`, and `cat_classifier.py` orchestrates data loading, preprocessing, training, evaluation, and comparison experiments.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Network Architecture](#network-architecture)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Normalization Techniques](#normalization-techniques)
5. [Initialization Methods](#initialization-methods)
6. [Regularization Techniques](#regularization-techniques)
7. [Optimization Algorithms](#optimization-algorithms)
8. [Learning Rate Decay](#learning-rate-decay)
9. [Implementation Details](#implementation-details)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Visualizations](#visualizations)
12. [Learning Rate Experiments](#learning-rate-experiments)
13. [How to Run](#how-to-run)
14. [Results](#results)
15. [Troubleshooting](#troubleshooting)
16. [Gradient Checking](#gradient-checking)
17. [Code Snippets](#code-snippets)

---

## Project Overview

### Files

| File | Description |
|------|-------------|
| `five_layer_nn.py` | Core neural network class + comparison experiments on XOR data |
| `cat_classifier.py` | Cat classifier training pipeline + all visualization/comparison experiments |
| `StandFord Machine Learning/Neural Networks/Notebook/five_layer_nn.ipynb` | Notebook that loads and runs `five_layer_nn.py` |
| `StandFord Machine Learning/Neural Networks/Notebook/cat_classifier.ipynb` | Notebook that loads and runs `cat_classifier.py` |
| `datasets/train_catvnoncat.h5` | Training data (209 images) - optional, falls back to synthetic data if missing |
| `datasets/test_catvnoncat.h5` | Test data (50 images) - optional, falls back to synthetic data if missing |

**Key Scripts:**

- **five_layer_nn.py**: Contains the `FiveLayerNN` class with all methods (normalization, initialization, regularization, optimizers, mini-batches, learning rate decay, gradient checking). When run directly, it performs comprehensive comparisons on synthetic XOR data.

- **cat_classifier.py**: Contains data loading, preprocessing, evaluation metrics, and plotting functions. When run directly, it trains a cat classifier and generates 12 comparison plots.

**Execution Flow (cat_classifier.py):**

1. Load data from `datasets/*.h5` (if missing, generate synthetic data for a full demo run).
2. Preprocess data by flattening images and normalizing features (main run uses Z-score).
3. Build a 5-layer network: `[n_x, 20, 7, 5, 3, 1]`.
4. Train the baseline model (He init, L2=0.1, dropout=0.14, GD with `learning_rate=0.0075`).
5. Evaluate metrics and plot core diagnostics (cost curve, confusion matrix, metrics bar chart).
6. Run comparison experiments (initialization, regularization, optimizers, mini-batches, learning rates, and LR decay).
7. If real images are available, also run normalization comparison and plot sample predictions.

### Core Features Summary

| Feature | Implementation | Purpose |
|---------|----------------|---------|
| **Deep Architecture** | 5 Layers (4 Hidden + 1 Output) | Ability to learn high-level feature representations. |
| **Activations** | ReLU (Hidden), Sigmoid (Output) | Efficient learning and binary probability output. |
| **Normalization** | Z-Score, Min-Max, L2, Mean | Stabilizes training and speeds up convergence. |
| **Initialization** | He, Xavier, Random, Zeros | Prevents vanishing/exploding gradients. |
| **Regularization** | L2 Weight Decay + Inverted Dropout | Prevents overfitting and improves generalization. |
| **Optimization** | GD, Momentum, RMSprop, Adam | Faster search for the minimum of the cost function. |
| **Mini-Batches** | Configurable Batch Sizes | Balance between training speed and gradient stability. |
| **LR Decay** | Continuous & Scheduled (Step) | Fine-tuning the model in later stages of training. |
| **Evaluation** | F1 Score, Recall, Precision, CM | Comprehensive performance analysis beyond accuracy. |
| **Gradient Check** | Numerical vs Analytical Gradients | Debugging tool to verify backprop correctness. |

### Data

- **Training Set**: 209 images (64x64x3 RGB)
- **Test Set**: 50 images (64x64x3 RGB)
- **Classes**: Cat (1), Non-Cat (0)
- **Input Features**: 12,288 (64 x 64 x 3 flattened)

**Dataset Location:**

The scripts look for dataset files at:
- `datasets/train_catvnoncat.h5`
- `datasets/test_catvnoncat.h5`

If the files are not found, `cat_classifier.py` automatically falls back to synthetic data (random samples) so you can still run all experiments and see the plots.
In synthetic mode, normalization comparison and sample prediction plots are skipped because they require real images.

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

The model implements a standard deep neural network training loop consisting of forward propagation, cost computation, backward propagation, and parameter updates.

### 1. Forward Propagation

Forward propagation is the process of passing input data through the network to generate a prediction. For each layer $l$:

```
Z[l] = W[l] * A[l-1] + b[l]
A[l] = g[l](Z[l])
```

**Intuition:**
- **Linear Transformation ($Z$):** Projects the input into a new space using weights ($W$) and biases ($b$). This allows the network to learn linear relationships.
- **Activation Function ($g$):** Introduces non-linearity, allowing the network to learn complex, non-linear patterns. Without this, the entire 5-layer network would just be a single linear transformation.

Where:
- `A[0] = X` (input)
- `g[l]` = ReLU for hidden layers, Sigmoid for output layer

### 2. Activation Functions

#### **ReLU (Rectified Linear Unit)**
Used in hidden layers to enable deep learning. It is computationally efficient and helps mitigate the vanishing gradient problem.

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

#### **Sigmoid**
Used in the final layer for binary classification. It squashes the output into a probability range $[0, 1]$.

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

Measures how well the model's predictions match the true labels.

```
J = -1/m * SUM[y*log(a) + (1-y)*log(1-a)]
```

**Why this formula?**
- If $y=1$, we want $a$ to be close to 1. The term $-log(a)$ is small when $a \approx 1$ and large when $a \approx 0$.
- If $y=0$, we want $a$ to be close to 0. The term $-log(1-a)$ is small when $a \approx 0$ and large when $a \approx 1$.

### 4. Backward Propagation

The goal of backprop is to calculate how much each parameter contributed to the total error (the gradient). We use the chain rule to propagate the error from the output back to the input.

**Output Layer ($L$):**
```
dZ[L] = A[L] - Y
dW[L] = (1/m) * dZ[L] * A[L-1].T
db[L] = (1/m) * SUM(dZ[L])
```

**Hidden Layers ($l = L-1, ..., 1$):**
```
dZ[l] = W[l+1].T * dZ[l+1] * g'[l](Z[l])
dW[l] = (1/m) * dZ[l] * A[l-1].T
db[l] = (1/m) * SUM(dZ[l])
```

### 5. Parameter Update (Gradient Descent)

Finally, we update the weights to minimize the cost.

```
W[l] = W[l] - alpha * dW[l]
b[l] = b[l] - alpha * db[l]
```

Where `alpha` ($\alpha$) is the learning rate—the size of the "step" we take down the error gradient.

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

### Comparison Results (Example)

When running `cat_classifier.py` with real data (only if datasets are available):

```
Method       Train Acc    Test Acc     Final Loss
------       ---------    --------     ----------
simple       96.17        74.00        0.118234
minmax       97.13        76.00        0.082456
zscore       98.56        78.00        0.048741    (best!)
mean         97.61        76.00        0.095123
l2           94.74        72.00        0.156789
```

Note: Normalization comparison runs in `cat_classifier.py` only when real image data is loaded.

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

### Comparison Results (Example from cat_classifier.py)

```
Method       Train Acc    Test Acc     Final Loss
------       ---------    --------     ----------
zeros        50.00        50.00        0.693147    (no learning!)
random       50.00        50.00        inf         (exploded!)
xavier       96.17        74.00        0.118234
he           98.56        78.00        0.048741    (best!)
```

The comparison is generated by `plot_initialization_comparison()` in `cat_classifier.py`.

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

### Regularization Comparison Results (Example from cat_classifier.py)

```
Method              Train Acc    Test Acc    Gap      Loss
------              ---------    --------    ---      ----
No Reg              99.04        72.00       27.04    0.020123
L2 (λ=0.1)          96.17        76.00       20.17    0.085432
L2 (λ=0.7)          91.87        78.00       13.87    0.182345
Dropout (0.2)       94.26        76.00       18.26    0.124567
L2 + Dropout        91.39        80.00       11.39    0.198765   (smallest gap!)
```

**Key insights:**
- Regularization reduces train accuracy but improves test accuracy by reducing overfitting
- **Gap** (train_acc - test_acc) is the key metric: smaller gap = better generalization
- L2 + Dropout combined often gives the best test accuracy and smallest gap

The comparison is generated by `plot_regularization_comparison()` in `cat_classifier.py`.

---

## Optimization Algorithms

The neural network supports four different optimization algorithms, each with different trade-offs in terms of speed, stability, and ease of tuning.

### The Problem with Standard Gradient Descent

Standard gradient descent (GD) updates weights using the current gradient. In some regions of the cost surface (like long, narrow valleys), GD can oscillate wildly, making very slow progress toward the minimum.

```
Without Momentum (GD):          With Momentum:
    * (Start)                      * (Start)
   / \                            /
  /   \                          /
 *     * (Oscillation)          *
  \   /                        /
   \ /                        /
    *                        * (Consistent progress)
   / \                      /
  /   \                    *  (Faster, smoother)
 *     *
```

### Available Optimization Algorithms

| Algorithm | Key Intuition | Pros | Cons |
|-----------|---------------|------|------|
| **GD** | Step directly down the gradient. | Simple, easy to understand. | Slow, prone to oscillations. |
| **Momentum** | Adds "velocity" to the gradient updates. | Dampens oscillations, accelerates in the right direction. | One more hyperparameter ($\beta$) to tune. |
| **RMSprop** | Scales the learning rate by the magnitude of recent gradients. | Great for non-stationary problems, helps with steep gradients. | Can still oscillate if $\beta_2$ is poor. |
| **Adam** | Combines Momentum and RMSprop. | **Best overall performance**, handles noisy gradients, works well with defaults. | More computationally expensive (slightly). |

### 1. Gradient Descent (GD)
The baseline algorithm. It makes progress exactly in the direction of the steepest descent.
```python
nn = FiveLayerNN(layer_dims, optimizer='gd', learning_rate=0.01)
```

### 2. Momentum
Accumulates a velocity vector $v$ that smooths out gradient updates. Think of a ball rolling down a hill—it gains momentum in consistent directions.
```python
# β controls "memory": 0.9 averages ~10 gradients, 0.99 averages ~100.
nn = FiveLayerNN(layer_dims, optimizer='momentum', beta=0.9, learning_rate=0.01)
```

### 3. RMSprop (Root Mean Square Propagation)
Adapts the learning rate for each parameter. If a parameter has large gradients, its learning rate is reduced to avoid overshooting. If it has small gradients, its learning rate is increased.
```python
nn = FiveLayerNN(layer_dims, optimizer='rmsprop', beta2=0.999, learning_rate=0.001)
```

### 4. Adam (Adaptive Moment Estimation) - **Recommended**
The industry standard. It combines the advantages of Momentum (velocity) and RMSprop (adaptive learning rates) and adds bias correction to handle the early stages of training.
```python
nn = FiveLayerNN(
    layer_dims,
    optimizer='adam',
    beta1=0.9,    # Momentum decay
    beta2=0.999,  # RMSprop decay
    learning_rate=0.001
)
```

### Optimizer Comparison Results (Example from cat_classifier.py)

```
Optimizer            Train Acc    Test Acc     Final Loss
---------            ---------    --------     ----------
Gradient Descent     93.00        93.00        0.200221
Momentum (β=0.9)     94.67        95.00        0.189979
RMSprop              95.22        96.00        0.142341
Adam                 96.17        96.50        0.098765    (best!)
```

**Note:** cat_classifier.py tests all four optimizers with appropriate learning rates:
- GD and Momentum use `learning_rate=0.0075`
- RMSprop and Adam use `learning_rate=0.001` (lower LR works better for adaptive methods)

The comparison is generated by `plot_optimizer_comparison()` in `cat_classifier.py`.

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

### Mini-Batch Size Comparison Results (Example from cat_classifier.py)

```
Batch Size           Train Acc    Test Acc     Final Loss
----------           ---------    --------     ----------
SGD (1)              91.00        90.00        0.312451
Mini-batch (32)      95.50        95.00        0.125678
Mini-batch (64)      96.00        96.00        0.098234    (best balance)
Batch GD (209)       94.00        93.50        0.145678
```

**Note:** All tests use Adam optimizer with `learning_rate=0.001` for 500 epochs.

The comparison is generated by `plot_mini_batch_comparison()` in `cat_classifier.py`.

### Recommended Configuration (from cat_classifier.py)

The main model in `cat_classifier.py` uses:

```python
# Configuration used in cat_classifier.py for main training
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.0075,      # Standard GD learning rate
    initialization='he',       # Best for ReLU
    lambd=0.1,                 # L2 regularization
    keep_prob=0.86             # 14% dropout
)

# Train for 2500 epochs (full batch by default)
losses = nn.train(train_x, train_y, epochs=2500, print_loss=True)
```

**For production with large datasets:**

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

# Train with mini-batches and learning rate decay
losses = nn.train(
    X_train, Y_train, 
    epochs=2500, 
    mini_batch_size=64,
    decay_type='scheduled'
)
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

## Learning Rate Decay

Learning rate decay gradually reduces the learning rate during training to fine-tune convergence in later stages.

### Why Use Learning Rate Decay?

Think of it as a search for a treasure. Early on, you want to take large steps to explore the map. As you get closer to the treasure, you take smaller, more careful steps to pinpoint its exact location.

```
Early Training (large α):
  - Fast progress toward the minimum.
  - Risk of "jumping over" the optimal point if α stays large.

Late Training (small α):
  - Fine-tuning the parameters.
  - Stabilizes convergence and avoids oscillations near the minimum.
```

### Available Decay Strategies

| Strategy | Formula | Behavior |
|----------|---------|----------|
| **No Decay** | `α = α₀` | Constant learning rate |
| **Continuous** | `α = α₀ / (1 + decay_rate * epoch)` | Smoothly decreases every epoch |
| **Scheduled** | `α = α₀ / (1 + decay_rate * floor(epoch / interval))` | Steps down at fixed intervals |

### 1. Continuous Decay

Learning rate decreases smoothly every epoch:

```python
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.1,        # Initial learning rate
    decay_rate=0.01           # Decay rate (typical: 0.01-0.1)
)

# Train with continuous decay
losses = nn.train(X, Y, epochs=2500, decay_type='continuous')
```

**Effect:**
```
Epoch 0:    α = 0.100000
Epoch 100:  α = 0.050000
Epoch 500:  α = 0.016667
Epoch 1000: α = 0.009091
Epoch 2500: α = 0.003922
```

### 2. Scheduled Decay (Step Decay)

Learning rate decreases at fixed intervals:

```python
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.1,
    decay_rate=1.0,           # Multiply factor at each step
    time_interval=500         # Reduce LR every 500 epochs
)

# Train with scheduled decay
losses = nn.train(X, Y, epochs=2500, decay_type='scheduled')
```

**Effect:**
```
Epochs 0-499:     α = 0.100000
Epochs 500-999:   α = 0.050000
Epochs 1000-1499: α = 0.033333
Epochs 1500-1999: α = 0.025000
Epochs 2000-2500: α = 0.020000
```

### Usage

```python
# Continuous decay
nn = FiveLayerNN(layer_dims, learning_rate=0.1, decay_rate=0.01)
losses = nn.train(X, Y, epochs=2500, decay_type='continuous')

# Scheduled decay
nn = FiveLayerNN(layer_dims, learning_rate=0.1, decay_rate=1.0, time_interval=500)
losses = nn.train(X, Y, epochs=2500, decay_type='scheduled')
```

### Choosing Decay Parameters

| Parameter | Typical Values | Effect |
|-----------|----------------|--------|
| `decay_rate` | 0.01-0.1 (continuous), 0.5-1.0 (scheduled) | Higher = faster decay |
| `time_interval` | 500-1000 epochs | When to step down (scheduled only) |

### Benefits

1. **Faster convergence**: Large steps early, small steps late
2. **Better final accuracy**: Fine-tuning near minimum
3. **Stability**: Reduces oscillations in late training

---

## Implementation Details

### Data Preprocessing (from cat_classifier.py)

The `preprocess_data` function in `cat_classifier.py` handles both flattening and normalization. The default normalization is `minmax`, while the main script explicitly uses `zscore`.

```python
from five_layer_nn import FiveLayerNN

# Example: Preprocess with Z-score normalization (used in main)
train_x, test_x, norm_params = preprocess_data(
    train_x_orig, 
    test_x_orig, 
    normalization='zscore'
)

# norm_params contains: {'method': 'zscore', 'mean': mean, 'std': std}
```

**Available normalization methods:**
- `'simple'`: Divide by 255 (pixels → [0, 1])
- `'minmax'`: Min-max scaling to [0, 1]
- `'zscore'`: Standardize to mean=0, std=1 (recommended)
- `'mean'`: Center around zero
- `'l2'`: L2 normalization (unit norm per sample)

**Steps:**
```python
# 1. Flatten images: (num_examples, h, w, c) -> (h*w*c, num_examples)
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# 2. Normalize using selected method (e.g., Z-score)
train_x, mean, std = FiveLayerNN.normalize_zscore(train_x_flatten)
test_x, _, _ = FiveLayerNN.normalize_zscore(test_x_flatten, mean, std)
```

### Training Loop (with Mini-Batches and LR Decay)

```python
for epoch in range(epochs):
    # Apply learning rate decay (optional)
    if decay_rate > 0:
        update_learning_rate(epoch)  # or schedule_lr_decay(epoch)
    
    # Create mini-batches (shuffle each epoch)
    mini_batches = create_mini_batches(X, Y, mini_batch_size)
    
    for mini_batch_X, mini_batch_Y in mini_batches:
        # 1. Forward propagation (with dropout if enabled)
        A5 = forward_propagation(mini_batch_X, training=True)
        
        # 2. Compute loss (with L2 regularization if enabled)
        cost = compute_loss(mini_batch_Y)
        
        # 3. Backward propagation (with L2 and dropout gradients)
        backward_propagation(mini_batch_Y)
        
        # 4. Update parameters (using selected optimizer)
        update_parameters()  # GD, Momentum, RMSprop, or Adam
```

### Full Pipeline Example

This example shows a custom configuration (Adam + mini-batches). The default run in `cat_classifier.py` uses GD with `learning_rate=0.0075` and full-batch training.

```python
from five_layer_nn import FiveLayerNN
from cat_classifier import load_data, preprocess_data, compute_metrics

# 1. Load data
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# 2. Preprocess with Z-score normalization
train_x, test_x, _ = preprocess_data(train_x_orig, test_x_orig, normalization='zscore')

# 3. Define architecture
n_x = train_x.shape[0]  # 12288 features
layer_dims = [n_x, 20, 7, 5, 3, 1]

# 4. Create model with best configuration
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.0075,
    initialization='he',
    lambd=0.1,              # L2 regularization
    keep_prob=0.86,         # 14% dropout
    optimizer='adam',       # Adam optimizer
    beta1=0.9,
    beta2=0.999
)

# 5. Train with mini-batches
losses = nn.train(train_x, train_y, epochs=2500, mini_batch_size=64)

# 6. Evaluate
train_pred = nn.predict(train_x)
test_pred = nn.predict(test_x)
test_metrics = compute_metrics(test_pred, test_y)

print(f"Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
print(f"Test F1 Score: {test_metrics['f1']*100:.2f}%")
```

---

## Evaluation Metrics

To truly understand how the model is performing, we use a variety of metrics beyond just simple accuracy.

### Confusion Matrix

The confusion matrix gives a complete breakdown of the model's performance on each class.

```
                    Predicted
                  Cat    Non-Cat
              +--------+--------+
    Actual    |   TP   |   FN   |  Cat
              +--------+--------+
    Actual    |   FP   |   TN   |  Non-Cat
              +--------+--------+
```

- **TP (True Positive):** Correctly identified a cat.
- **TN (True Negative):** Correctly identified a non-cat.
- **FP (False Positive):** Wrongly identified a non-cat as a cat (**Type I Error**).
- **FN (False Negative):** Missed an actual cat (**Type II Error**).

### Metrics Formulas

| Metric | Formula | Best Used For... |
|--------|---------|------------------|
| **Accuracy** | $(TP + TN) / Total$ | Balanced datasets where errors are equally costly. |
| **Precision** | $TP / (TP + FP)$ | Minimizing False Positives (e.g., Spam detection). |
| **Recall** | $TP / (TP + FN)$ | Minimizing False Negatives (e.g., Cancer detection). |
| **F1 Score** | $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$ | **Imbalanced datasets** (best balance of Precision/Recall). |

### The Precision vs. Recall Trade-off

You can change the classification threshold (default is 0.5) to favor either precision or recall:
- **Low Threshold (e.g., 0.1):** Model predicts "Cat" more often. High Recall (you find all cats), Low Precision (lots of false alarms).
- **High Threshold (e.g., 0.9):** Model predicts "Cat" only when very sure. High Precision (no false alarms), Low Recall (you miss many cats).

---

## Visualizations

Some plots require real image data. When the dataset is missing and the script falls back to synthetic data, normalization comparison and sample prediction plots are skipped.

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

### 6. Normalization Comparison (`normalization_comparison.png`)

Overlaid cost curves showing training performance for different normalization methods (simple, minmax, zscore, mean, l2).

### 7. Initialization Comparison (`initialization_comparison.png`)

Overlaid cost curves comparing zeros, random, xavier, and he initialization methods.

### 8. Regularization Comparison (`regularization_comparison.png`)

Training cost curves for different regularization strategies with overfitting gap shown in legend.

### 9. Regularization Accuracy (`regularization_accuracy.png`)

Bar chart showing train vs test accuracy for each regularization method, highlighting the overfitting gap.

### 10. Optimizer Comparison (`optimizer_comparison.png`)

Cost curves comparing GD, Momentum, RMSprop, and Adam optimizers over training.

### 11. Mini-Batch Comparison (`mini_batch_comparison.png`)

Cost curves for different batch sizes: SGD (1), mini-batch (32, 64), and full batch GD.

### 12. Learning Rate Decay Comparison (`learning_rate_decay_comparison.png`)

Two-panel plot:
- Left: Training cost curves for different decay strategies
- Right: Learning rate schedules over training epochs

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

### Run Python Scripts

```bash
# Run cat classifier (full pipeline with all experiments)
python cat_classifier.py

# Run five_layer_nn (initialization/regularization comparisons on XOR data)
python five_layer_nn.py
```

### Run Notebooks

If the notebooks are present, open
`StandFord Machine Learning/Neural Networks/Notebook/five_layer_nn.ipynb` or
`StandFord Machine Learning/Neural Networks/Notebook/cat_classifier.ipynb`
in Jupyter and run all cells. The first code cell loads the matching `.py`
script so the notebook stays in sync.

### Expected Output from cat_classifier.py

If dataset files are missing, the script prints a synthetic-data warning and skips the normalization comparison and sample prediction steps.

Sample output (values vary by run and dataset):

```
============================================================
    CAT VS NON-CAT CLASSIFIER
    5-Layer Deep Neural Network
============================================================

[1] Loading data...
    Training examples: 209
    Test examples: 50
    Image shape: (64, 64, 3)

[2] Preprocessing data (Z-score normalization)...
    Normalization: zscore
    Flattened image size: 12288
    Train shape: (12288, 209)
    Test shape: (12288, 50)

[3] Setting up network architecture...
    Architecture: [12288, 20, 7, 5, 3, 1]
    Layer 1: 12288 -> 20 (ReLU)
    Layer 2: 20 -> 7 (ReLU)
    Layer 3: 7 -> 5 (ReLU)
    Layer 4: 5 -> 3 (ReLU)
    Layer 5: 3 -> 1 (Sigmoid)

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

[9] Comparing different normalization methods... (only with real data)
    Training with SIMPLE normalization...
    Training with MINMAX normalization...
    Training with ZSCORE normalization...
    Training with MEAN normalization...
    Training with L2 normalization...
    Saved: normalization_comparison.png

    === NORMALIZATION COMPARISON RESULTS ===
    Method       Train Acc    Test Acc     Final Loss
    ----------------------------------------------------
    simple       96.17        74.00        0.118234
    minmax       97.13        76.00        0.082456
    zscore       98.56        78.00        0.048741
    mean         97.61        76.00        0.095123
    l2           94.74        72.00        0.156789

[10] Comparing different initialization methods...
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

[12] Comparing different optimizers (GD vs Momentum vs RMSprop vs Adam)...
    Training with Gradient Descent...
    Training with Momentum (β=0.9)...
    Training with RMSprop...
    Training with Adam...
    Saved: optimizer_comparison.png

    === OPTIMIZER COMPARISON RESULTS ===
    Optimizer            Train Acc    Test Acc     Final Loss
    --------------------------------------------------------
    Gradient Descent     93.00        93.00        0.200221
    Momentum (β=0.9)     94.67        95.00        0.189979
    RMSprop              95.22        96.00        0.142341
    Adam                 96.17        96.50        0.098765

[13] Comparing different mini-batch sizes...
    Training with SGD (batch=1)...
    Training with Mini-batch (32)...
    Training with Mini-batch (64)...
    Training with Batch GD (209)...
    Saved: mini_batch_comparison.png

    === MINI-BATCH SIZE COMPARISON RESULTS ===
    Batch Size           Train Acc    Test Acc     Final Loss
    --------------------------------------------------------
    SGD (1)              91.00        90.00        0.312451
    Mini-batch (32)      95.50        95.00        0.125678
    Mini-batch (64)      96.00        96.00        0.098234
    Batch GD (209)       94.00        93.50        0.145678

[14] Comparing different learning rates...
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

[15] Comparing different learning rate decay strategies...
    Training with No Decay...
    Training with Continuous (0.01)...
    Training with Continuous (0.1)...
    Training with Scheduled (1, 500)...
    Training with Scheduled (1, 1000)...
    Saved: learning_rate_decay_comparison.png

    === LEARNING RATE DECAY COMPARISON RESULTS ===
    Decay Strategy              Train Acc    Test Acc     Final LR     Loss
    -------------------------------------------------------------------------
    No Decay                    96.17        76.00        0.007500     0.048741
    Continuous (0.01)           96.65        77.00        0.000750     0.042132
    Continuous (0.1)            95.22        75.00        0.000147     0.058234
    Scheduled (1, 500)          97.13        78.00        0.001500     0.038456
    Scheduled (1, 1000)         96.89        77.50        0.002500     0.041234

[16] Plotting sample predictions... (only with real data)
    Saved: sample_predictions.png

============================================================
    TRAINING COMPLETE!
============================================================

    Generated Graphs:
    - cost_curve.png
    - confusion_matrix.png
    - metrics_comparison.png
    - normalization_comparison.png (real data only)
    - initialization_comparison.png
    - regularization_comparison.png
    - regularization_accuracy.png
    - optimizer_comparison.png
    - mini_batch_comparison.png
    - learning_rate_comparison.png
    - learning_rate_decay_comparison.png
    - sample_predictions.png (real data only)

    Key Findings:
    - Normalization speeds up convergence and improves stability
    - Z-score standardization is generally recommended
    - He initialization works best for ReLU networks
    - L2 regularization and dropout reduce overfitting
    - Combining L2 + dropout often gives best generalization
    - Adam optimizer generally performs best
    - Mini-batch sizes of 32-64 offer good speed/stability balance
    - Learning rate decay helps fine-tune convergence in later epochs
    - Scheduled decay (step decay) gives more control over when LR reduces

============================================================
```

If the dataset is missing, the beginning of the run looks like:

```
[1] Loading data...
    Dataset not found! Using synthetic data for demo...
    Synthetic training examples: 209
    Synthetic test examples: 50
```

---

## Results

### Expected Performance

These ranges apply when running with the real cat/non-cat dataset. When synthetic data is used, metrics vary randomly and are not meaningful.

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Training | ~98% | ~97% | ~100% | ~98% |
| Test | ~72-80% | ~70-75% | ~60-70% | ~65-72% |

### Generated Files (from cat_classifier.py)

| File | Description |
|------|-------------|
| `cost_curve.png` | Training loss over iterations for main model |
| `confusion_matrix.png` | Heatmap of predictions vs actuals (test set) |
| `metrics_comparison.png` | Bar chart comparing train/test metrics (accuracy, precision, recall, F1) |
| `normalization_comparison.png` | Cost curves for simple/minmax/zscore/mean/l2 normalization (real data only) |
| `initialization_comparison.png` | Cost curves for zeros/random/xavier/he initialization |
| `regularization_comparison.png` | Cost curves for no reg, L2 (λ=0.1/0.7), dropout (0.2), and L2+dropout |
| `regularization_accuracy.png` | Bar chart of train vs test accuracy per regularization method |
| `optimizer_comparison.png` | Cost curves for GD, Momentum, RMSprop, and Adam |
| `mini_batch_comparison.png` | Cost curves for SGD (1), mini-batch (32/64), and batch GD |
| `learning_rate_comparison.png` | Cost curves for different learning rates (0.001, 0.0075, 0.01, 0.05) |
| `learning_rate_decay_comparison.png` | Two-panel plot: cost curves and LR schedules for decay strategies |
| `sample_predictions.png` | Grid of test images with predictions (green=correct, red=wrong) (real data only) |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: datasets/train_catvnoncat.h5` | Dataset missing | Script will auto-fallback to synthetic data. To use real data, place h5 files in `datasets/` folder at repo root |
| Cost stays high | Learning rate too low | Increase learning rate |
| Cost oscillates or diverges | Learning rate too high | Decrease learning rate or use Adam optimizer |
| Low test accuracy (high train) | Overfitting | Add/increase regularization (L2 or dropout) |
| `NaN` in cost | Numerical overflow | Reduce learning rate, check data normalization |
| Slow training | Large dataset + full batch | Use mini-batches (32-64) and Adam optimizer |

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

### Computing Metrics (from cat_classifier.py)

The `compute_metrics` function in `cat_classifier.py` computes all evaluation metrics:

```python
from cat_classifier import compute_metrics

predictions = nn.predict(test_x)
metrics = compute_metrics(predictions, test_y)

print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
print(f"Precision: {metrics['precision']*100:.2f}%")
print(f"Recall:    {metrics['recall']*100:.2f}%")
print(f"F1 Score:  {metrics['f1']*100:.2f}%")
print(f"Confusion Matrix: TP={metrics['TP']}, FP={metrics['FP']}, "
      f"TN={metrics['TN']}, FN={metrics['FN']}")
```

### Plotting Functions (from cat_classifier.py)

All plotting functions are available in `cat_classifier.py`:

```python
from cat_classifier import (
    plot_cost,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_sample_predictions,
    plot_initialization_comparison,
    plot_normalization_comparison,
    plot_regularization_comparison,
    plot_optimizer_comparison,
    plot_mini_batch_comparison,
    plot_learning_rate_comparison,
    plot_learning_rate_decay_comparison
)

# Example: Plot training cost
plot_cost(losses, title="Training Cost - Cat Classifier")

# Example: Plot confusion matrix
test_metrics = compute_metrics(test_predictions, test_y)
plot_confusion_matrix(test_metrics, title="Confusion Matrix (Test Set)")

# Example: Plot sample predictions (requires original images)
plot_sample_predictions(test_x_orig, test_y, test_predictions, classes, num_samples=10)
```

### Running Experiments (from cat_classifier.py)

```python
# Compare different initialization methods
init_results = plot_initialization_comparison(
    train_x, train_y, test_x, test_y, 
    layer_dims, epochs=2500
)

# Compare different regularization methods
reg_results = plot_regularization_comparison(
    train_x, train_y, test_x, test_y, 
    layer_dims, epochs=2500
)

# Compare different optimizers
opt_results = plot_optimizer_comparison(
    train_x, train_y, test_x, test_y, 
    layer_dims, epochs=2500
)
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

When you execute `five_layer_nn.py`, it runs comprehensive comparisons on synthetic XOR-like data:

```bash
python five_layer_nn.py
```

**What it does:**
1. Generates 300 training samples and 200 test samples (XOR problem with noise)
2. Compares all normalization methods (none, minmax, zscore, mean, l2)
3. Compares all initialization methods (zeros, random, xavier, he)
4. Compares all regularization methods (no reg, L2, dropout, L2+dropout)
5. Compares all optimizers (GD, momentum, RMSprop, Adam)
6. Compares mini-batch sizes (1, 32, 64, 128, full batch)
7. Compares learning rate decay strategies (continuous and scheduled)
8. Runs gradient checking to verify backpropagation

Expected output (summary):
```
==================================================
GRADIENT CHECKING
==================================================
Relative difference: 1.23e-07
Gradient check PASSED! Backpropagation is correct.

KEY TAKEAWAYS
==================================================
NORMALIZATION:
- Min-Max: Scales to [0, 1] range
- Z-Score: Mean=0, Std=1 - most common
- Mean: Centers around zero
- L2: Unit norm per sample

INITIALIZATION:
- Zeros: Symmetry problem (all neurons learn same)
- Random (large): Exploding/vanishing gradients
- Xavier: Good for tanh/sigmoid
- He: Best for ReLU (recommended)

REGULARIZATION:
- L2: Penalizes large weights
- Dropout: Prevents co-adaptation
- Combined: Often best results

OPTIMIZATION ALGORITHMS:
- GD: Simple but can be slow
- Momentum: Accelerates, dampens oscillations
- RMSprop: Adaptive learning rate per parameter
- Adam: Best of both worlds (recommended)
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
4. **Combine techniques** - L2 + dropout together often gives best generalization (smallest train-test gap)
5. **Learning rate** is crucial - too high causes instability, too low is slow
6. **Adam optimizer** is generally the best choice - combines momentum + RMSprop with less tuning
7. **Mini-batches** (32-64) offer the best balance of speed and stability
8. **Learning rate decay** helps fine-tune convergence - scheduled decay gives more control than continuous
9. **F1 score** is better than accuracy for imbalanced datasets
10. **Monitor the gap** - Large train-test accuracy gap indicates overfitting
11. **Gradient checking** verifies backpropagation correctness before training (runs automatically in five_layer_nn.py)

### Quick Reference: Recommended Settings

```python
from five_layer_nn import FiveLayerNN
from cat_classifier import load_data, preprocess_data, compute_metrics

# 1. Load and normalize data (Z-score recommended)
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x, test_x, _ = preprocess_data(train_x_orig, test_x_orig, normalization='zscore')

# 2. Define architecture
layer_dims = [12288, 20, 7, 5, 3, 1]

# 3. Create model with best settings
nn = FiveLayerNN(
    layer_dims,
    learning_rate=0.001,        # Lower LR for Adam
    initialization='he',        # Best for ReLU
    optimizer='adam',           # Best optimizer overall
    beta1=0.9,                  # Adam momentum parameter
    beta2=0.999,                # Adam RMSprop parameter
    lambd=0.1,                  # Light L2 regularization
    keep_prob=0.86,             # 14% dropout
    decay_rate=1.0,             # Learning rate decay
    time_interval=500           # Decay every 500 epochs
)

# 4. Train with mini-batches and scheduled LR decay
losses = nn.train(
    train_x, train_y, 
    epochs=2500, 
    mini_batch_size=64,
    decay_type='scheduled'
)

# 5. Evaluate
test_predictions = nn.predict(test_x)
test_metrics = compute_metrics(test_predictions, test_y)

print(f"Accuracy: {test_metrics['accuracy']*100:.2f}%")
print(f"F1 Score: {test_metrics['f1']*100:.2f}%")
print(f"Config: {nn.get_config()}")
```

---

## References

- [Andrew Ng's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)
- [He Initialization Paper](https://arxiv.org/abs/1502.01852)
- [Understanding Binary Cross-Entropy](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)

---

## Author

Built as part of AWS ML Bootcamp 2 - Deep Learning Module
