# FIXED VERSION - Cell 3

# Gradient descent to update weights and bias
learning_rate = 0.01
num_iterations = 1000

def train_logistic_regression(X, y, weights, bias, learning_rate, num_iterations):
    m = y.shape[0]
    loss_history = []
    
    for i in range(num_iterations):
        # Forward pass
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        
        # Compute loss
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(1/m) * np.sum(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        loss_history.append(loss)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        
        # Update parameters
        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss:.4f}")
    
    return weights, bias, loss_history

# Train the model
final_weights, final_bias, loss_history = train_logistic_regression(
    X_train, y_train_array, weights, bias, learning_rate, num_iterations
)

# Compute final loss using UPDATED weights
z_final = np.dot(X_train, final_weights) + final_bias  # ✅ Use final_weights
y_pred_final = sigmoid(z_final)
y_pred_clipped = np.clip(y_pred_final, epsilon, 1 - epsilon)
final_loss = -(1/m) * np.sum(y_train_array * np.log(y_pred_clipped) + 
                              (1 - y_train_array) * np.log(1 - y_pred_clipped))

print(f"\nFinal loss after training: {final_loss:.4f}")
print(f"Final weights: {final_weights}")
print(f"Final bias: {final_bias}")

# Plot loss curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.grid(True)
plt.show()
