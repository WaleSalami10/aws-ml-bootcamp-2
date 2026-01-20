import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

# Generate sample email data (word count vs spam probability)
np.random.seed(42)
word_counts = np.random.uniform(0, 100, 200).reshape(-1, 1)
spam_labels = (word_counts.flatten() > 50).astype(int) + np.random.binomial(1, 0.2, 200)
spam_labels = np.clip(spam_labels, 0, 1)

# Train classification model
model = LogisticRegression()
model.fit(word_counts, spam_labels)

# Generate predictions for smooth curve
x_line = np.linspace(0, 100, 100).reshape(-1, 1)
y_prob = model.predict_proba(x_line)[:, 1]

# Create chart
plt.figure(figsize=(10, 6))
colors = ['green' if label == 0 else 'red' for label in spam_labels]
labels = ['Not Spam' if label == 0 else 'Spam' for label in spam_labels]
plt.scatter(word_counts, spam_labels, c=colors, alpha=0.6, s=50)
plt.plot(x_line, y_prob, color='blue', linewidth=2, label='Classification Boundary')
plt.xlabel('Suspicious Word Count')
plt.ylabel('Spam Probability')
plt.title('Email Spam Classification')
plt.ylim(-0.1, 1.1)
plt.legend(['Classification Curve', 'Not Spam', 'Spam'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spam_classification_chart.png', dpi=300, bbox_inches='tight')
plt.close()