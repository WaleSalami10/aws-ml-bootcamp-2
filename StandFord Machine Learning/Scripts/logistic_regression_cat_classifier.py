"""
Logistic Regression implementation for Cat vs Non-Cat classification.
Built from scratch using only NumPy for educational purposes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class LogisticRegressionCatClassifier:
    """
    Logistic Regression classifier implemented from scratch for binary classification.
    Specifically designed for cat vs non-cat image classification.
    """
    
    def __init__(self, learning_rate=0.01, num_iterations=2000):
        """
        Initialize the logistic regression classifier.
        
        Parameters:
        learning_rate (float): Step size for gradient descent
        num_iterations (int): Number of training iterations
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.costs = []
        
    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        Parameters:
        z (numpy.ndarray): Linear combination of features
        
        Returns:
        numpy.ndarray: Sigmoid activation values
        """
        # Clip z to prevent overflow in exponential
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, num_features):
        """
        Initialize weights and bias.
        
        Parameters:
        num_features (int): Number of input features
        """
        # Initialize weights to small random values
        self.weights = np.random.randn(num_features, 1) * 0.01
        self.bias = 0.0
        
    def forward_propagation(self, X):
        """
        Perform forward propagation to compute predictions.
        
        Parameters:
        X (numpy.ndarray): Input features of shape (num_features, num_samples)
        
        Returns:
        numpy.ndarray: Predicted probabilities
        """
        # Linear combination: z = w^T * x + b
        z = np.dot(self.weights.T, X) + self.bias
        
        # Apply sigmoid activation
        A = self.sigmoid(z)
        
        return A
    
    def compute_cost(self, A, Y):
        """
        Compute the logistic regression cost function.
        
        Parameters:
        A (numpy.ndarray): Predicted probabilities
        Y (numpy.ndarray): True labels
        
        Returns:
        float: Cost value
        """
        m = Y.shape[1]  # Number of samples
        
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        A = np.clip(A, epsilon, 1 - epsilon)
        
        # Cross-entropy cost function
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        
        return cost
    
    def backward_propagation(self, A, X, Y):
        """
        Compute gradients using backward propagation.
        
        Parameters:
        A (numpy.ndarray): Predicted probabilities
        X (numpy.ndarray): Input features
        Y (numpy.ndarray): True labels
        
        Returns:
        tuple: Gradients for weights and bias
        """
        m = Y.shape[1]  # Number of samples
        
        # Compute gradients
        dw = 1/m * np.dot(X, (A - Y).T)
        db = 1/m * np.sum(A - Y)
        
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        Update parameters using gradient descent.
        
        Parameters:
        dw (numpy.ndarray): Weight gradients
        db (float): Bias gradient
        """
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db
    
    def fit(self, X, y):
        """
        Train the logistic regression model.
        
        Parameters:
        X (numpy.ndarray): Training features of shape (num_samples, num_features)
        y (numpy.ndarray): Training labels of shape (num_samples,)
        """
        # Reshape data for matrix operations
        X = X.T  # Shape: (num_features, num_samples)
        Y = y.reshape(1, -1)  # Shape: (1, num_samples)
        
        # Initialize parameters
        num_features = X.shape[0]
        self.initialize_parameters(num_features)
        
        print("Starting training...")
        print(f"Dataset: {X.shape[1]} samples, {X.shape[0]} features")
        print(f"Learning rate: {self.learning_rate}, Iterations: {self.num_iterations}")
        print("-" * 50)
        
        # Training loop
        for i in range(self.num_iterations):
            # Forward propagation
            A = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(A, Y)
            self.costs.append(cost)
            
            # Backward propagation
            dw, db = self.backward_propagation(A, X, Y)
            
            # Update parameters
            self.update_parameters(dw, db)
            
            # Print progress every 200 iterations
            if i % 200 == 0:
                print(f"Cost after iteration {i}: {cost:.6f}")
        
        print(f"Training completed! Final cost: {self.costs[-1]:.6f}")
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        X (numpy.ndarray): Input features of shape (num_samples, num_features)
        
        Returns:
        tuple: (predictions, probabilities)
        """
        X = X.T  # Transpose for matrix operations
        
        # Forward propagation
        A = self.forward_propagation(X)
        
        # Convert probabilities to binary predictions
        predictions = (A > 0.5).astype(int).flatten()
        probabilities = A.flatten()
        
        return predictions, probabilities
    
    def evaluate(self, X, y):
        """
        Evaluate model performance.
        
        Parameters:
        X (numpy.ndarray): Test features
        y (numpy.ndarray): True labels
        
        Returns:
        dict: Performance metrics
        """
        predictions, probabilities = self.predict(X)
        
        # Calculate metrics
        accuracy = np.mean(predictions == y)
        
        # Confusion matrix components
        tp = np.sum((predictions == 1) & (y == 1))  # True positives
        fp = np.sum((predictions == 1) & (y == 0))  # False positives
        tn = np.sum((predictions == 0) & (y == 0))  # True negatives
        fn = np.sum((predictions == 0) & (y == 1))  # False negatives
        
        # Calculate precision, recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }

def load_data():
    """Load the cat classification dataset"""
    data_dir = "StandFord Machine Learning/Data"
    
    # Load training data
    train_df = pd.read_csv(f"{data_dir}/cat_classification_train.csv")
    train_X = train_df.drop('label', axis=1).values
    train_y = train_df['label'].values
    
    # Load test data
    test_df = pd.read_csv(f"{data_dir}/cat_classification_test.csv")
    test_X = test_df.drop('label', axis=1).values
    test_y = test_df['label'].values
    
    # Normalize features to [0, 1] range
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    
    return train_X, train_y, test_X, test_y

def plot_training_progress(costs, save_path):
    """Plot the cost function during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(costs, 'b-', linewidth=2)
    plt.title('Logistic Regression Training Progress', fontsize=16)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost (Cross-Entropy Loss)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictions_analysis(model, test_X, test_y, save_path):
    """Plot prediction analysis"""
    predictions, probabilities = model.predict(test_X)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Logistic Regression Prediction Analysis', fontsize=16)
    
    # 1. Probability distribution
    axes[0, 0].hist(probabilities[test_y == 1], bins=30, alpha=0.7, label='Cats', color='orange')
    axes[0, 0].hist(probabilities[test_y == 0], bins=30, alpha=0.7, label='Non-Cats', color='blue')
    axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Probability Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confusion Matrix
    metrics = model.evaluate(test_X, test_y)
    cm = metrics['confusion_matrix']
    confusion_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
    
    im = axes[0, 1].imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    axes[0, 1].set_title('Confusion Matrix')
    tick_marks = np.arange(2)
    axes[0, 1].set_xticks(tick_marks)
    axes[0, 1].set_yticks(tick_marks)
    axes[0, 1].set_xticklabels(['Non-Cat', 'Cat'])
    axes[0, 1].set_yticklabels(['Non-Cat', 'Cat'])
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, confusion_matrix[i, j], ha="center", va="center", fontsize=14)
    
    # 3. ROC-like curve (simplified)
    thresholds = np.linspace(0, 1, 100)
    tpr_values = []
    fpr_values = []
    
    for threshold in thresholds:
        pred_thresh = (probabilities > threshold).astype(int)
        tp = np.sum((pred_thresh == 1) & (test_y == 1))
        fp = np.sum((pred_thresh == 1) & (test_y == 0))
        tn = np.sum((pred_thresh == 0) & (test_y == 0))
        fn = np.sum((pred_thresh == 0) & (test_y == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    axes[1, 0].plot(fpr_values, tpr_values, 'b-', linewidth=2)
    axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curve')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Performance metrics
    axes[1, 1].axis('off')
    metrics_text = f"""
    Performance Metrics:
    
    Accuracy: {metrics['accuracy']:.4f}
    Precision: {metrics['precision']:.4f}
    Recall: {metrics['recall']:.4f}
    F1-Score: {metrics['f1_score']:.4f}
    
    Confusion Matrix:
    True Negatives: {cm['tn']}
    False Positives: {cm['fp']}
    False Negatives: {cm['fn']}
    True Positives: {cm['tp']}
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the complete logistic regression pipeline"""
    print("=" * 60)
    print("LOGISTIC REGRESSION FOR CAT vs NON-CAT CLASSIFICATION")
    print("=" * 60)
    
    # Create output directory
    output_dir = "StandFord Machine Learning/Supervised Learning/Logistic Regression/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\n1. Loading dataset...")
    train_X, train_y, test_X, test_y = load_data()
    
    print(f"Training set: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    print(f"Test set: {test_X.shape[0]} samples, {test_X.shape[1]} features")
    print(f"Feature range: [{train_X.min():.3f}, {train_X.max():.3f}]")
    
    # Initialize and train model
    print("\n2. Training logistic regression model...")
    model = LogisticRegressionCatClassifier(learning_rate=0.005, num_iterations=2000)
    model.fit(train_X, train_y)
    
    # Evaluate on training set
    print("\n3. Evaluating on training set...")
    train_metrics = model.evaluate(train_X, train_y)
    print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Training Precision: {train_metrics['precision']:.4f}")
    print(f"Training Recall: {train_metrics['recall']:.4f}")
    print(f"Training F1-Score: {train_metrics['f1_score']:.4f}")
    
    # Evaluate on test set
    print("\n4. Evaluating on test set...")
    test_metrics = model.evaluate(test_X, test_y)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    
    # Plot training progress
    plot_training_progress(model.costs, f"{output_dir}/training_progress.png")
    print(f"Training progress plot saved to {output_dir}/training_progress.png")
    
    # Plot prediction analysis
    plot_predictions_analysis(model, test_X, test_y, f"{output_dir}/prediction_analysis.png")
    print(f"Prediction analysis plot saved to {output_dir}/prediction_analysis.png")
    
    # Model summary
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Algorithm: Logistic Regression (Binary Classification)")
    print(f"Dataset: Cat vs Non-Cat Images (64x64x3 = 12,288 features)")
    print(f"Training samples: {len(train_X)}")
    print(f"Test samples: {len(test_X)}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Iterations: {model.num_iterations}")
    print(f"Final training cost: {model.costs[-1]:.6f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    
    # Feature importance (top weights)
    print(f"\nModel parameters:")
    print(f"Weights shape: {model.weights.shape}")
    print(f"Bias: {model.bias:.6f}")
    print(f"Weight range: [{model.weights.min():.6f}, {model.weights.max():.6f}]")
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()