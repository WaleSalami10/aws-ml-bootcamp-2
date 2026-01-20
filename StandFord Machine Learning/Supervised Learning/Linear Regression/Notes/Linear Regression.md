![alt text](images/house_prediction_regression.png)

Linear regression is a supervised learning algorithm that is used to predict a continuous output variable (y) based on one or more input variables (x). The goal of linear regression is to find the best-fitting line or hyperplane that minimizes the difference between the predicted values and the actual values.

Terminologies
1. Features (x): The input variables used to predict the output variable. In the house price prediction example, features could include the size of the house, number of bedrooms, location, etc.
2. Target (y): The output variable that we are trying to predict. In the house price prediction example, the target is the price of the house.
3. Coefficients (w): The weights assigned to each feature in the linear equation. These coefficients determine the importance of each feature in predicting the target variable.
4. Intercept (b): The value of the target variable when all the features are equal to zero. It represents the baseline value of the target variable.
5. (m) is the number of training examples
6. (x,y) = single training example
7. (n) is the number of features
8. (i) is the index of the training example
9. (y-hat) is the symbol used to represent the predicted value of the target variable in linear regression. 
    
print("x^(i) represents the input features of the i-th training example")
print("y^(i) represents the target/output value of the i-th training example")
(x^i, y^i) = training example of the i-th row.
The linear regression equation can be represented as:
y = wx + b
f(x) = y-hat
f(x) = wx + b
y-hat = wx + b

This is the formular for linear regression with 1 variable (univariate linear regression)
For multiple variables, the equation can be represented as:
y-hat = w1x1 + w2x2 + w3x3 +... + wnxn + b

import numpy as np

# Generate sample data for house price prediction


# Generate features
n_samples = 100
size = np.random.normal(2000, 500, n_samples)  # House size in sqft
bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
age = np.random.randint(0, 50, n_samples)      # Age of house in years

# Generate target prices using a linear combination with some noise
# Price = w1*size + w2*bedrooms + w3*age + b + noise
w1, w2, w3 = 100, 50000, -2000  # Coefficients
b = 100000                       # Intercept/bias
noise = np.random.normal(0, 25000, n_samples)

prices = (w1 * size + 
         w2 * bedrooms + 
         w3 * age + 
         b + 
         noise)

# Model formula: y = w1*x1 + w2*x2 + w3*x3 + b
# where:
# y = house price (target)
# x1 = house size in sqft
# x2 = number of bedrooms  
# x3 = age of house
# w1, w2, w3 = coefficients
# b = intercept

# Create feature matrix X and target vector y
X = np.column_stack((size, bedrooms, age))
y = prices

# Create a function to calculate the y-hat
def predict(X, weights, bias):
    return np.dot(X, weights) + bias

# Example usage
weights = np.array([w1, w2, w3])
bias = b
y_hat = predict(X, weights, bias)
print(y_hat)
print(y)

# plot the graph using the price and the size in square feet
import matplotlib.pyplot as plt
plt.scatter(range(len(y)), y, color='blue', label='Actual Prices')
plt.scatter(range(len(y_hat)), y_hat, color='red', label='Predicted Prices')
plt.xlabel('size')
plt.ylabel('House Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()

This is assuming we know our w and b parameters.


 b is also refered to as the y intercept because that is where it crosses the y-axis
![alt text](images/y-intercept.png)

# WHAT IS COST FUNCTION ?
The cost function is a mathematical function that measures the difference between the predicted values (y-hat) and the actual values (y) of the target variable. The goal of linear regression is to minimize the cost function, which is typically done using optimization algorithms such as gradient descent.
The most common cost function used in linear regression is the Mean Squared Error (MSE), which is calculated as follows:

J= MSE = (1/m) * Σ(y^(i) - y-hat^(i))^2
MSE = (1/m)* Σ(y^(i) - f(x)^(i))^2  # knowing y-hat = f(x)
where:
m = number of training examples
y^(i) = actual value of the target variable for the i-th training example
y-hat^(i) = predicted value of the target variable for the i-th training example
The MSE measures the average squared difference between the predicted values and the actual values. The lower
the MSE, the better the model is at predicting the target variable.

![alt text](images/cost_function_plot.png)

# Explain the correlation of cost function j and w from the graph
The graph shows the relationship between the cost function (J) and the weight (w) in a linear regression model. The cost function is represented on the vertical axis, while the weight is represented on the horizontal axis.
As we can see from the graph, the cost function is a convex function, which means that it has a single minimum point. The minimum point represents the optimal weight (w) that minimizes the cost function (J). In other words, it is the weight that results in the best fit line for the given data.
As we move away from the optimal weight in either direction (increasing or decreasing w), the
cost function (J) increases. This indicates that the model's predictions are getting worse as we move away from the optimal weight.
The shape of the graph also indicates that the cost function is sensitive to changes in the weight.
This means that small changes in the weight can result in significant changes in the cost function. Therefore, it is important to choose the optimal weight carefully to ensure that the model is accurate and reliable.

![alt text](images/cost_function_3d_plot.png)

# Explain the above graph
The graph shows a 3D representation of the cost function (J) in relation to weights (w) bias (b) in a linear regression model. The cost function is represented on the vertical axis, while w and b  are represented on the horizontal axes.
As we can see from the graph, the cost function is a convex surface, which means that it has a single minimum point. The minimum point represents the optimal combination of weights (w) and bias (b) that minimizes the cost function (J). In other words, it is the combination that results in the best fit line for the given data.

# Write and explain a python function to calculate cost function
import numpy as np
def calculate_cost(y, y_hat):
    return np.mean((y - y_hat) ** 2) / 2
This function calculates the Mean Squared Error (MSE) cost function for linear regression.
The function takes two arguments:
y: The actual values of the target variable.
y_hat: The predicted values of the target variable.
The function calculates the MSE by taking the mean of the squared differences between the actual values (
y) and the predicted values (y_hat). The result is divided by 2 to simplify the derivative calculation during optimization.

# Example usage
cost = calculate_cost(y, y_hat)
print(f"Cost: {cost}")  # Target with noise

# Explain Gradient descent
Gradient descent is an optimization algorithm used to minimize the cost function in machine learning models, including linear regression. The goal of gradient descent is to find the optimal values of the model parameters (weights and bias) that minimize the cost function.
The basic idea behind gradient descent is to iteratively adjust the model parameters in the direction of the
steepest descent of the cost function. This is done by calculating the gradient (partial derivatives) of the cost function with respect to each parameter and updating the parameters accordingly.
The gradient descent algorithm can be summarized in the following steps:
1. Initialize the model parameters (weights and bias) with random values.
2. Calculate the predicted values (y-hat) using the current model parameters.
3. Calculate the cost function (J) using the actual values (y) and the predicted
values (y-hat).
4. Calculate the gradients (partial derivatives) of the cost function with respect to each parameter.
5. Update the model parameters by subtracting a fraction of the gradients from the current parameter values

# Gradient descent algorithm
import numpy as np
def gradient_descent(X, y, weights, bias, learning_rate, n_iterations):
    m = len(y)  # number of training examples
    for _ in range(n_iterations):
        y_hat = np.dot(X, weights) + bias  # predicted values
        error = y_hat - y  # error between predicted and actual values
        
        # Calculate gradients
        dw = (1/m) * np.dot(X.T, error)  # gradient w.r.t. weights
        db = (1/m) * np.sum(error)         # gradient w.r.t. bias
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
    return weights, bias
# Example usage
learning_rate = 0.01
n_iterations = 1000
weights = np.array([0.0, 0.0, 0.0])
bias = 0.0
weights, bias = gradient_descent(X, y, weights, bias, learning_rate, n_iterations)
y_hat = predict(X, weights, bias)
print(f"Optimized Weights: {weights}, Optimized Bias: {bias}")

# Explain the gradient descent formular
The gradient descent formula is used to update the model parameters (weights and bias) in order to minimize the cost function. The formulas for updating the weights and bias are as follows:
weights = weights - learning_rate * dw
bias = bias - learning_rate * db

where:
weights: The current values of the model weights (coefficients).
bias: The current value of the model bias (intercept).
learning_rate: A hyperparameter that determines the step size for each update. It controls how much
the model parameters are adjusted during each iteration of gradient descent.
dw: The gradient (partial derivative) of the cost function with respect to the weights.
db: The gradient (partial derivative) of the cost function with respect to the bias.
The gradients (dw and db) are calculated based on the error between the predicted values (y-hat) and the actual values (y). The gradients indicate the direction and magnitude of the steepest ascent of the cost function. By subtracting a fraction of the gradients (scaled by the learning rate) from the current parameter values, we move in the direction of steepest descent, thereby reducing the cost function.

# Write and explain Mathematical expression of dw and db
The mathematical expressions for the gradients (dw and db) in linear regression are as follows:
dw = (1/m) * Σ(x^(i) * (y-hat^(i) - y^(i)))
db = (1/m) * Σ(y-hat^(i) - y^(i))

where:
dw: The gradient (partial derivative) of the cost function with respect to the weights.
db: The gradient (partial derivative) of the cost function with respect to the bias.
m: The number of training examples.
x^(i): The input features of the i-th training example.
y^(i): The actual value of the target variable for the i-th training example.
y-hat^(i): The predicted value of the target variable for the i-th training example
The expressions for dw and db are derived from the Mean Squared Error (MSE) cost function. The gradients indicate how much the cost function changes with respect to small changes in the weights and bias. By calculating these gradients, we can determine the direction in which to adjust the weights and bias to minimize the cost function.

# Full mathematical expression of gradient descent
weights = weights - learning_rate * (1/m) * Σ(x^(i) * (y-hat^(i) - y^(i)))
bias = bias - learning_rate * (1/m) * Σ(y-hat^(i) - y^(i))
where:
weights: The current values of the model weights (coefficients).
bias: The current value of the model bias (intercept).
learning_rate: A hyperparameter that determines the step size for each update.
dw: The gradient (partial derivative) of the cost function with respect to the weights.
db: The gradient (partial derivative) of the cost function with respect to the bias.
m: The number of training examples.
x^(i): The input features of the i-th training example.
y^(i): The actual value of the target variable for the i-th training example.
y-hat^(i): The predicted value of the target variable for the i-th training example

# Example usage of cost function
cost = calculate_cost(y, y_hat)
print(f"Cost: {cost}")  # Target with noise

# Example usage of gradient descent function
weights, bias = gradient_descent(X, y, weights, bias)
print(f"Updated weights: {weights}, Updated bias: {bias}")








