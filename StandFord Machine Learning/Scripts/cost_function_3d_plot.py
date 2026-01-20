import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
m = len(x)

# Range of w and b values
w_values = np.linspace(-1, 3, 50)
b_values = np.linspace(-2, 4, 50)
W, B = np.meshgrid(w_values, b_values)

# Calculate cost J(w,b) for each combination
J = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        y_hat = W[i, j] * x + B[i, j]
        J[i, j] = (1 / (2 * m)) * np.sum((y_hat - y) ** 2)

# Create 3D surface plot
fig = plt.figure(figsize=(12, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(W, B, J, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('J(w,b)')
ax1.set_title('Cost Function J(w,b) - 3D Surface')

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(W, B, J, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('w')
ax2.set_ylabel('b')
ax2.set_title('Cost Function J(w,b) - Contour Plot')

plt.tight_layout()
plt.savefig('cost_function_3d_plot.png', dpi=300, bbox_inches='tight')
plt.close()
