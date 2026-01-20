# Logistic Regression for Cat vs Non-Cat Image Classification

## Overview
Logistic regression is a statistical method used for binary classification problems. In this implementation, we'll classify images as either containing a cat (1) or not containing a cat (0).

## Mathematical Foundation

### Sigmoid Function
The sigmoid function maps any real number to a value between 0 and 1, making it perfect for probability estimation:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $z = w^T x + b$ (linear combination of features)
- $w$ = weight vector
- $x$ = input features (flattened image pixels)
- $b$ = bias term

### Cost Function
For logistic regression, we use the cross-entropy loss function:

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$

Where:
- $m$ = number of training examples
- $y^{(i)}$ = true label (0 or 1)
- $\hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$ = predicted probability

### Gradient Descent Updates
To minimize the cost function, we compute gradients and update parameters:

$$\frac{\partial J}{\partial w} = \frac{1}{m} X (A - Y)^T$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (a^{(i)} - y^{(i)})$$

Parameter updates:
$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

Where $\alpha$ is the learning rate.

## Implementation Steps

1. **Data Preparation**: Generate synthetic image data and flatten to feature vectors
2. **Parameter Initialization**: Initialize weights and bias to small random values
3. **Forward Propagation**: Compute predictions using sigmoid function
4. **Cost Computation**: Calculate cross-entropy loss
5. **Backward Propagation**: Compute gradients
6. **Parameter Updates**: Update weights and bias using gradient descent
7. **Prediction**: Use trained model to classify new images

## Key Concepts

### Feature Engineering for Images
- Images are represented as pixel intensity matrices
- For RGB images: height × width × 3 channels
- Flatten to 1D vector: (height × width × 3,) features per image
- Normalize pixel values to [0,1] range for better convergence

### Binary Classification Decision Boundary
- If $\hat{y} \geq 0.5$: predict cat (1)
- If $\hat{y} < 0.5$: predict non-cat (0)
- The decision boundary occurs where $w^T x + b = 0$

### Evaluation Metrics
- **Accuracy**: Percentage of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall