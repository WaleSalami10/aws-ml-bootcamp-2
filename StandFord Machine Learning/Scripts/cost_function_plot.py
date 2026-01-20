import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
m = len(x)
b = 0  # Fixed intercept

# Range of w values to test
w_values = np.linspace(-2, 6, 100)
j_values = []

# Calculate cost J(w) for each w
for w in w_values:
    y_hat = w * x + b
    j = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)
    j_values.append(j)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(w_values, j_values, linewidth=2, color='blue')
plt.xlabel('w (weight parameter)')
plt.ylabel('J(w) (cost function)')
plt.title('Cost Function J(w) vs Parameter w')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cost_function_plot.png', dpi=300, bbox_inches='tight')
plt.close()
