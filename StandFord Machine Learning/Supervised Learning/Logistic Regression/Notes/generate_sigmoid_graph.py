import numpy as np
import matplotlib.pyplot as plt

# Generate z values
z = np.linspace(-10, 10, 200)

# Calculate sigmoid
sigmoid = 1 / (1 + np.exp(-z))

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid, 'b-', linewidth=2, label='σ(z) = 1/(1+e^(-z))')

# Add reference lines
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary (0.5)')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='gray', linestyle='-', alpha=0.3)

# Mark key points
plt.plot(0, 0.5, 'ro', markersize=10, label='z=0, σ(z)=0.5')
plt.plot(-5, 1/(1+np.exp(5)), 'go', markersize=8)
plt.plot(5, 1/(1+np.exp(-5)), 'go', markersize=8)

# Labels and formatting
plt.xlabel('z (linear combination)', fontsize=12)
plt.ylabel('σ(z) (probability)', fontsize=12)
plt.title('Sigmoid Function - S-shaped Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.ylim(-0.1, 1.1)
plt.xlim(-10, 10)

# Add annotations
plt.text(-7, 0.1, 'Class 0\n(Negative)', fontsize=10, ha='center', color='blue')
plt.text(7, 0.9, 'Class 1\n(Positive)', fontsize=10, ha='center', color='blue')
plt.text(0, 0.55, 'Threshold', fontsize=9, ha='center', color='red')

plt.tight_layout()
plt.savefig('sigmoid_curve.png', dpi=300, bbox_inches='tight')
print("Sigmoid graph saved as 'sigmoid_curve.png'")
plt.close()
