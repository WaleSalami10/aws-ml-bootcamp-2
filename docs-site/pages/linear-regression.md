# Linear Regression

Linear regression is a fundamental supervised learning algorithm that models the relationship between a dependent variable and independent variables by fitting a linear equation to observed data.

## 📖 Mathematical Foundation

### Hypothesis Function
The linear regression hypothesis is:

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T x$$

### Cost Function
We use the Mean Squared Error (MSE) cost function:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### Gradient Descent Update Rules
Parameters are updated using:

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

Where the partial derivatives are:
- $$\frac{\partial}{\partial \theta_0} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$
- $$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$$

## 🔧 Implementation Features

### Core Algorithm
- **Gradient Descent**: Iterative optimization with learning rate control
- **Cost Tracking**: Monitor convergence with cost history
- **Numerical Stability**: Epsilon clipping for robust computation

### Regularization Techniques

#### Ridge Regression (L2)
Adds L2 penalty to prevent overfitting:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

#### Lasso Regression (L1)
Uses L1 penalty for feature selection:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

#### Elastic Net
Combines L1 and L2 penalties:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda_1}{m} \sum_{j=1}^{n} |\theta_j| + \frac{\lambda_2}{2m} \sum_{j=1}^{n} \theta_j^2$$

## 📊 Datasets & Applications

### House Price Prediction
- **Features**: Size, bedrooms, age, location score
- **Target**: Price prediction
- **Use Case**: Real estate valuation

### Visualization Outputs
- Cost function convergence plots
- Prediction vs actual scatter plots
- Regularization path visualization
- Feature importance analysis

## 📁 Available Resources

### Notebooks
1. **Linear_regression.ipynb** - Basic implementation
2. **Linear_regression_regularization.ipynb** - Advanced techniques

### Documentation
- **Linear_Regression_Complete_Guide.ipynb** - Interactive theory and code
- Mathematical derivations with step-by-step explanations

### Generated Data
- **house_prices.csv** - Synthetic housing dataset
- Configurable parameters for experimentation

## 🎯 Key Learning Outcomes

- Understand the mathematical foundation of linear regression
- Implement gradient descent from scratch
- Apply regularization techniques to prevent overfitting
- Visualize algorithm behavior and convergence
- Handle real-world data preprocessing challenges

---

*Implementation follows educational best practices with NumPy-only core algorithms and comprehensive mathematical documentation.*