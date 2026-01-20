# Logistic Regression

Logistic regression is a statistical method used for binary classification problems. It uses the logistic function to model the probability that an instance belongs to a particular category.

## 📖 Mathematical Foundation

### Sigmoid Function
The logistic regression uses the sigmoid activation function:

$$g(z) = \frac{1}{1 + e^{-z}}$$

### Hypothesis Function
The hypothesis function becomes:

$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

### Cost Function
We use the logistic cost function (cross-entropy):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$$

### Gradient Descent Update Rules
The gradient for logistic regression is:

$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

Parameter updates:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

## 🔧 Implementation Features

### Core Algorithm
- **Sigmoid Activation**: Probabilistic output between 0 and 1
- **Cross-Entropy Loss**: Appropriate cost function for classification
- **Gradient Descent**: Iterative parameter optimization
- **Decision Boundary**: Threshold-based classification

### Regularization Techniques

#### Ridge Logistic Regression (L2)
Adds L2 penalty to the cost function:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

#### Lasso Logistic Regression (L1)
Uses L1 penalty for feature selection:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))] + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

#### Elastic Net Logistic Regression
Combines L1 and L2 penalties:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))] + \frac{\lambda_1}{m} \sum_{j=1}^{n} |\theta_j| + \frac{\lambda_2}{2m} \sum_{j=1}^{n} \theta_j^2$$

## 📊 Datasets & Applications

### Tumor Classification
- **Features**: Tumor size, patient age, blood markers, genetic factors
- **Target**: Malignant (1) vs Benign (0)
- **Use Case**: Medical diagnosis support

### Performance Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## 📊 Visualization Outputs

### Algorithm Behavior
- Cost function convergence plots
- Decision boundary visualization
- Sigmoid function plots
- Probability distribution analysis

### Model Performance
- Confusion matrix heatmaps
- ROC curves and AUC scores
- Precision-recall curves
- Feature importance plots

## 📁 Available Resources

### Notebooks
1. **Logistic_regression.ipynb** - Basic implementation
2. **Logistic_regression_regularization.ipynb** - Advanced techniques

### Documentation
- **Logistic_Regression_Complete_Guide.ipynb** - Interactive theory and code
- Mathematical derivations with detailed explanations

### Generated Data
- **tumor_data.csv** - Synthetic medical classification dataset
- Balanced classes for robust evaluation

## 🎯 Key Learning Outcomes

- Understand the sigmoid function and its properties
- Implement cross-entropy cost function from scratch
- Apply regularization to prevent overfitting in classification
- Evaluate classification performance with multiple metrics
- Visualize decision boundaries and model behavior

## 🔍 Advanced Topics

### Numerical Stability
- Epsilon clipping to prevent log(0) errors
- Gradient clipping for stable convergence
- Feature scaling for improved optimization

### Model Interpretation
- Coefficient interpretation as log-odds ratios
- Feature importance analysis
- Regularization path visualization

---

*Implementation emphasizes mathematical understanding with NumPy-only core algorithms and comprehensive performance evaluation.*