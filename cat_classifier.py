import numpy as np
import h5py
import matplotlib.pyplot as plt
from five_layer_nn import FiveLayerNN


# ==================== Data Loading ====================

def load_data():
    """
    Load the cat vs non-cat dataset.

    Returns:
    train_x_orig -- training images, shape (num_examples, height, width, channels)
    train_y -- training labels, shape (1, num_train_examples)
    test_x_orig -- test images, shape (num_examples, height, width, channels)
    test_y -- test labels, shape (1, num_test_examples)
    classes -- list of class names
    """
    # Load training data
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:])

    # Load test data
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:])

    # Get class names
    classes = np.array(test_dataset["list_classes"][:])

    # Reshape labels to (1, num_examples)
    train_y = train_y.reshape((1, train_y.shape[0]))
    test_y = test_y.reshape((1, test_y.shape[0]))

    return train_x_orig, train_y, test_x_orig, test_y, classes


def preprocess_data(train_x_orig, test_x_orig, normalization='minmax'):
    """
    Preprocess images: flatten and normalize.

    Arguments:
    train_x_orig -- training images, shape (num_examples, height, width, channels)
    test_x_orig -- test images, shape (num_examples, height, width, channels)
    normalization -- normalization method: 'minmax', 'zscore', 'mean', 'l2', or 'simple'

    Returns:
    train_x -- flattened and normalized training images, shape (num_features, num_examples)
    test_x -- flattened and normalized test images, shape (num_features, num_examples)
    norm_params -- normalization parameters (for applying to new data)
    """
    # Flatten images: (num_examples, h, w, c) -> (h*w*c, num_examples)
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    norm_params = {}

    if normalization == 'simple':
        # Simple normalization: divide by 255 (0-255 -> 0-1)
        train_x = train_x_flatten / 255.
        test_x = test_x_flatten / 255.
        norm_params = {'method': 'simple'}

    elif normalization == 'minmax':
        # Min-Max normalization using FiveLayerNN method
        train_x, X_min, X_max = FiveLayerNN.normalize_minmax(train_x_flatten)
        test_x, _, _ = FiveLayerNN.normalize_minmax(test_x_flatten, X_min, X_max)
        norm_params = {'method': 'minmax', 'min': X_min, 'max': X_max}

    elif normalization == 'zscore':
        # Z-score standardization
        train_x, mean, std = FiveLayerNN.normalize_zscore(train_x_flatten)
        test_x, _, _ = FiveLayerNN.normalize_zscore(test_x_flatten, mean, std)
        norm_params = {'method': 'zscore', 'mean': mean, 'std': std}

    elif normalization == 'mean':
        # Mean normalization
        train_x, mean = FiveLayerNN.normalize_mean(train_x_flatten)
        test_x, _ = FiveLayerNN.normalize_mean(test_x_flatten, mean)
        norm_params = {'method': 'mean', 'mean': mean}

    elif normalization == 'l2':
        # L2 normalization
        train_x = FiveLayerNN.normalize_l2(train_x_flatten)
        test_x = FiveLayerNN.normalize_l2(test_x_flatten)
        norm_params = {'method': 'l2'}

    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    return train_x, test_x, norm_params


# ==================== Evaluation Metrics ====================

def compute_metrics(predictions, Y):
    """
    Compute precision, recall, F1 score, and confusion matrix.

    Arguments:
    predictions -- predicted labels (0 or 1), shape (1, m)
    Y -- true labels (0 or 1), shape (1, m)

    Returns:
    metrics -- dictionary with precision, recall, f1, confusion matrix
    """
    predictions = predictions.flatten()
    Y = Y.flatten()

    # True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((predictions == 1) & (Y == 1))
    FP = np.sum((predictions == 1) & (Y == 0))
    TN = np.sum((predictions == 0) & (Y == 0))
    FN = np.sum((predictions == 0) & (Y == 1))

    # Precision: Of all predicted cats, how many are actually cats?
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall: Of all actual cats, how many did we correctly identify?
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score: Harmonic mean of precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN
    }


# ==================== Plotting Functions ====================

def plot_cost(losses, title="Cost over Training"):
    """Plot the training cost curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.xlabel('Iterations (per 100)', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('cost_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    Saved: cost_curve.png")


def plot_learning_rate_comparison(train_x, train_y, layer_dims, learning_rates, epochs=2500):
    """
    Train models with different learning rates and compare.

    Arguments:
    train_x -- training data
    train_y -- training labels
    layer_dims -- network architecture
    learning_rates -- list of learning rates to try
    epochs -- number of training epochs
    """
    plt.figure(figsize=(12, 8))

    results = {}

    for lr in learning_rates:
        print(f"\n    Training with learning_rate = {lr}...")
        nn = FiveLayerNN(layer_dims, learning_rate=lr, initialization='he')
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False)

        # Sample losses every 100 iterations for plotting
        sampled_losses = losses[::1]  # Already sampled in train()

        results[lr] = {
            'losses': sampled_losses,
            'final_loss': losses[-1],
            'model': nn
        }

        plt.plot(sampled_losses, label=f'lr = {lr}', linewidth=2)

    plt.xlabel('Iterations (per 100)', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Learning Rate Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: learning_rate_comparison.png")

    return results


def plot_initialization_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500):
    """
    Train models with different initialization methods and compare.
    """
    plt.figure(figsize=(12, 8))

    initializations = ['zeros', 'random', 'xavier', 'he']
    results = {}

    for init in initializations:
        print(f"\n    Training with {init.upper()} initialization...")
        np.random.seed(42)
        nn = FiveLayerNN(layer_dims, learning_rate=0.0075, initialization=init)
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(train_x, train_y)
        test_acc = nn.accuracy(test_x, test_y)

        results[init] = {
            'losses': losses,
            'final_loss': losses[-1] if losses else float('inf'),
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': nn
        }

        plt.plot(losses, label=f'{init} (test: {test_acc:.1f}%)', linewidth=2)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Initialization Method Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('initialization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: initialization_comparison.png")

    return results


def plot_normalization_comparison(train_x_orig, train_y, test_x_orig, test_y, layer_dims, epochs=2500):
    """
    Train models with different normalization methods and compare.
    """
    normalizations = ['simple', 'minmax', 'zscore', 'mean', 'l2']
    results = {}

    plt.figure(figsize=(12, 8))

    for norm in normalizations:
        print(f"\n    Training with {norm.upper()} normalization...")
        np.random.seed(42)

        # Preprocess with specific normalization
        train_x, test_x, _ = preprocess_data(train_x_orig, test_x_orig, normalization=norm)

        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.0075,
            initialization='he'
        )
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(train_x, train_y)
        test_acc = nn.accuracy(test_x, test_y)

        results[norm] = {
            'losses': losses,
            'final_loss': losses[-1],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': nn
        }

        plt.plot(losses, label=f'{norm} (test: {test_acc:.1f}%)', linewidth=2)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Normalization Method Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: normalization_comparison.png")

    return results


def plot_regularization_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500):
    """
    Train models with different regularization methods and compare.
    """
    configs = [
        {'name': 'No Reg', 'lambd': 0.0, 'keep_prob': 1.0},
        {'name': 'L2 (λ=0.1)', 'lambd': 0.1, 'keep_prob': 1.0},
        {'name': 'L2 (λ=0.7)', 'lambd': 0.7, 'keep_prob': 1.0},
        {'name': 'Dropout (0.2)', 'lambd': 0.0, 'keep_prob': 0.8},
        {'name': 'L2 + Dropout', 'lambd': 0.1, 'keep_prob': 0.86},
    ]

    results = {}

    # Plot training curves
    plt.figure(figsize=(12, 8))

    for cfg in configs:
        print(f"\n    Training with {cfg['name']}...")
        np.random.seed(42)
        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.0075,
            initialization='he',
            lambd=cfg['lambd'],
            keep_prob=cfg['keep_prob']
        )
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(train_x, train_y)
        test_acc = nn.accuracy(test_x, test_y)

        results[cfg['name']] = {
            'losses': losses,
            'final_loss': losses[-1],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc,
            'model': nn,
            'config': cfg
        }

        plt.plot(losses, label=f"{cfg['name']} (gap: {train_acc - test_acc:.1f}%)", linewidth=2)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Regularization Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('regularization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: regularization_comparison.png")

    # Plot bar chart comparing train vs test accuracy
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(results.keys())
    train_accs = [results[n]['train_acc'] for n in names]
    test_accs = [results[n]['test_acc'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_accs, width, label='Train', color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='coral')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Regularization: Train vs Test Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('regularization_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    Saved: regularization_accuracy.png")

    return results


def plot_optimizer_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500):
    """
    Train models with different optimization algorithms and compare.
    """
    configs = [
        {'name': 'Gradient Descent', 'optimizer': 'gd', 'lr': 0.0075},
        {'name': 'Momentum (β=0.9)', 'optimizer': 'momentum', 'beta': 0.9, 'lr': 0.0075},
        {'name': 'RMSprop', 'optimizer': 'rmsprop', 'beta2': 0.999, 'lr': 0.001},
        {'name': 'Adam', 'optimizer': 'adam', 'beta1': 0.9, 'beta2': 0.999, 'lr': 0.001},
    ]

    results = {}

    # Plot training curves
    plt.figure(figsize=(12, 8))

    for cfg in configs:
        print(f"\n    Training with {cfg['name']}...")
        np.random.seed(42)
        nn = FiveLayerNN(
            layer_dims,
            learning_rate=cfg.get('lr', 0.0075),
            initialization='he',
            optimizer=cfg['optimizer'],
            beta=cfg.get('beta', 0.9),
            beta1=cfg.get('beta1', 0.9),
            beta2=cfg.get('beta2', 0.999)
        )
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(train_x, train_y)
        test_acc = nn.accuracy(test_x, test_y)

        results[cfg['name']] = {
            'losses': losses,
            'final_loss': losses[-1],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': nn,
            'config': cfg
        }

        plt.plot(losses, label=f"{cfg['name']} (test: {test_acc:.1f}%)", linewidth=2)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Optimizer Comparison (GD vs Momentum vs RMSprop vs Adam)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: optimizer_comparison.png")

    return results


def plot_mini_batch_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=500):
    """
    Train models with different mini-batch sizes and compare.
    """
    m = train_x.shape[1]
    batch_configs = [
        {'name': 'SGD (batch=1)', 'batch_size': 1},
        {'name': 'Mini-batch (32)', 'batch_size': 32},
        {'name': 'Mini-batch (64)', 'batch_size': 64},
        {'name': f'Batch GD ({m})', 'batch_size': m},
    ]

    results = {}

    # Plot training curves
    plt.figure(figsize=(12, 8))

    for cfg in batch_configs:
        print(f"\n    Training with {cfg['name']}...")
        np.random.seed(42)
        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.001,
            initialization='he',
            optimizer='adam'
        )
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False,
                         mini_batch_size=cfg['batch_size'])

        train_acc = nn.accuracy(train_x, train_y)
        test_acc = nn.accuracy(test_x, test_y)

        results[cfg['name']] = {
            'losses': losses,
            'final_loss': losses[-1],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'model': nn,
            'batch_size': cfg['batch_size']
        }

        plt.plot(losses, label=f"{cfg['name']} (test: {test_acc:.1f}%)", linewidth=2)

    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Mini-Batch Size Comparison (using Adam optimizer)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('mini_batch_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: mini_batch_comparison.png")

    return results


def plot_learning_rate_decay_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500):
    """
    Train models with different learning rate decay strategies and compare.
    """
    configs = [
        {'name': 'No Decay', 'decay_rate': 0.0, 'time_interval': 1000, 'decay_type': 'scheduled'},
        {'name': 'Continuous (0.01)', 'decay_rate': 0.01, 'time_interval': 1000, 'decay_type': 'continuous'},
        {'name': 'Continuous (0.1)', 'decay_rate': 0.1, 'time_interval': 1000, 'decay_type': 'continuous'},
        {'name': 'Scheduled (1, 500)', 'decay_rate': 1.0, 'time_interval': 500, 'decay_type': 'scheduled'},
        {'name': 'Scheduled (1, 1000)', 'decay_rate': 1.0, 'time_interval': 1000, 'decay_type': 'scheduled'},
    ]

    results = {}

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for cfg in configs:
        print(f"\n    Training with {cfg['name']}...")
        np.random.seed(42)
        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.0075,
            initialization='he',
            decay_rate=cfg['decay_rate'],
            time_interval=cfg['time_interval']
        )
        losses = nn.train(train_x, train_y, epochs=epochs, print_loss=False,
                         decay_type=cfg['decay_type'])

        train_acc = nn.accuracy(train_x, train_y)
        test_acc = nn.accuracy(test_x, test_y)

        results[cfg['name']] = {
            'losses': losses,
            'final_loss': losses[-1],
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_lr': nn.learning_rate,
            'lr_history': nn.learning_rate_history if hasattr(nn, 'learning_rate_history') else [],
            'model': nn,
            'config': cfg
        }

        axes[0].plot(losses, label=f"{cfg['name']} (test: {test_acc:.1f}%)", linewidth=2)

        # Plot learning rate history
        if hasattr(nn, 'learning_rate_history') and len(nn.learning_rate_history) > 0:
            axes[1].plot(nn.learning_rate_history, label=cfg['name'], linewidth=2)

    # Cost plot
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Cost', fontsize=12)
    axes[0].set_title('Training Cost with Different LR Decay Strategies', fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Over Time', fontsize=14)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_rate_decay_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n    Saved: learning_rate_decay_comparison.png")

    return results


def plot_confusion_matrix(metrics, title="Confusion Matrix"):
    """Plot confusion matrix as a heatmap."""
    cm = np.array([[metrics['TN'], metrics['FP']],
                   [metrics['FN'], metrics['TP']]])

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    plt.colorbar()

    classes = ['Non-Cat', 'Cat']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16)

    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    Saved: confusion_matrix.png")


def plot_sample_predictions(test_x_orig, test_y, predictions, classes, num_samples=10):
    """Plot sample images with predictions."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    indices = np.random.choice(test_x_orig.shape[0], num_samples, replace=False)

    for idx, ax in enumerate(axes):
        i = indices[idx]
        ax.imshow(test_x_orig[i])

        pred_label = classes[int(predictions[0, i])].decode("utf-8")
        true_label = classes[int(test_y[0, i])].decode("utf-8")

        color = 'green' if predictions[0, i] == test_y[0, i] else 'red'
        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    Saved: sample_predictions.png")


def plot_metrics_bar(train_metrics, test_metrics):
    """Plot comparison of metrics between train and test."""
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    train_values = [train_metrics['accuracy'], train_metrics['precision'],
                    train_metrics['recall'], train_metrics['f1']]
    test_values = [test_metrics['accuracy'], test_metrics['precision'],
                   test_metrics['recall'], test_metrics['f1']]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='steelblue')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='coral')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    Saved: metrics_comparison.png")


# ==================== Main ====================

if __name__ == "__main__":

    print("=" * 60)
    print("    CAT VS NON-CAT CLASSIFIER")
    print("    5-Layer Deep Neural Network")
    print("=" * 60)

    # ============ Step 1: Load Data ============
    print("\n[1] Loading data...")
    try:
        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
        print(f"    Training examples: {train_x_orig.shape[0]}")
        print(f"    Test examples: {test_x_orig.shape[0]}")
        print(f"    Image shape: {train_x_orig.shape[1:]}")

        # ============ Step 2: Preprocess Data ============
        print("\n[2] Preprocessing data (Z-score normalization)...")
        train_x, test_x, norm_params = preprocess_data(train_x_orig, test_x_orig, normalization='zscore')
        print(f"    Normalization: {norm_params['method']}")
        print(f"    Flattened image size: {train_x.shape[0]}")
        print(f"    Train shape: {train_x.shape}")
        print(f"    Test shape: {test_x.shape}")

        has_real_data = True

    except FileNotFoundError:
        print("    Dataset not found! Using synthetic data for demo...")
        np.random.seed(1)
        n_x = 12288
        m_train = 209
        m_test = 50

        train_x = np.random.randn(n_x, m_train)
        train_y = (np.random.rand(1, m_train) > 0.5).astype(int)
        test_x = np.random.randn(n_x, m_test)
        test_y = (np.random.rand(1, m_test) > 0.5).astype(int)
        classes = np.array([b'non-cat', b'cat'])
        train_x_orig = None
        test_x_orig = None

        print(f"    Synthetic training examples: {m_train}")
        print(f"    Synthetic test examples: {m_test}")

        has_real_data = False

    # ============ Step 3: Define Network Architecture ============
    print("\n[3] Setting up network architecture...")
    n_x = train_x.shape[0]
    layer_dims = [n_x, 20, 7, 5, 3, 1]

    print(f"    Architecture: {layer_dims}")
    print(f"    Layer 1: {layer_dims[0]} -> {layer_dims[1]} (ReLU)")
    print(f"    Layer 2: {layer_dims[1]} -> {layer_dims[2]} (ReLU)")
    print(f"    Layer 3: {layer_dims[2]} -> {layer_dims[3]} (ReLU)")
    print(f"    Layer 4: {layer_dims[3]} -> {layer_dims[4]} (ReLU)")
    print(f"    Layer 5: {layer_dims[4]} -> {layer_dims[5]} (Sigmoid)")

    # ============ Step 4: Create and Train Network ============
    print("\n[4] Training network (He init, L2=0.1, dropout=0.14)...")
    print("-" * 60)

    nn = FiveLayerNN(
        layer_dims,
        learning_rate=0.0075,
        initialization='he',
        lambd=0.1,
        keep_prob=0.86
    )
    losses = nn.train(train_x, train_y, epochs=2500, print_loss=True)

    print("-" * 60)
    print(f"    Config: {nn.get_config()}")

    # ============ Step 5: Plot Cost Curve ============
    print("\n[5] Plotting cost curve...")
    plot_cost(losses, title="Training Cost - Cat vs Non-Cat Classifier")

    # ============ Step 6: Evaluate Performance with Metrics ============
    print("\n[6] Computing evaluation metrics...")

    train_predictions = nn.predict(train_x)
    test_predictions = nn.predict(test_x)

    train_metrics = compute_metrics(train_predictions, train_y)
    test_metrics = compute_metrics(test_predictions, test_y)

    print("\n    === TRAINING SET METRICS ===")
    print(f"    Accuracy:  {train_metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {train_metrics['precision']*100:.2f}%")
    print(f"    Recall:    {train_metrics['recall']*100:.2f}%")
    print(f"    F1 Score:  {train_metrics['f1']*100:.2f}%")

    print("\n    === TEST SET METRICS ===")
    print(f"    Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"    Precision: {test_metrics['precision']*100:.2f}%")
    print(f"    Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"    F1 Score:  {test_metrics['f1']*100:.2f}%")

    # ============ Step 7: Plot Confusion Matrix ============
    print("\n[7] Plotting confusion matrix...")
    plot_confusion_matrix(test_metrics, title="Confusion Matrix (Test Set)")

    # ============ Step 8: Plot Metrics Comparison ============
    print("\n[8] Plotting metrics comparison...")
    plot_metrics_bar(train_metrics, test_metrics)

    # ============ Step 9: Normalization Comparison ============
    if has_real_data and train_x_orig is not None:
        print("\n[9] Comparing different normalization methods...")
        norm_results = plot_normalization_comparison(train_x_orig, train_y, test_x_orig, test_y, layer_dims, epochs=2500)

        print("\n    === NORMALIZATION COMPARISON RESULTS ===")
        print(f"    {'Method':<12} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
        print("    " + "-" * 48)
        for method, result in norm_results.items():
            print(f"    {method:<12} {result['train_acc']:<12.2f} {result['test_acc']:<12.2f} {result['final_loss']:<12.6f}")

    # ============ Step 10: Initialization Comparison ============
    print("\n[10] Comparing different initialization methods...")
    init_results = plot_initialization_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500)

    print("\n    === INITIALIZATION COMPARISON RESULTS ===")
    print(f"    {'Method':<12} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("    " + "-" * 48)
    for init, result in init_results.items():
        print(f"    {init:<12} {result['train_acc']:<12.2f} {result['test_acc']:<12.2f} {result['final_loss']:<12.6f}")

    # ============ Step 11: Regularization Comparison ============
    print("\n[11] Comparing different regularization methods...")
    reg_results = plot_regularization_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500)

    print("\n    === REGULARIZATION COMPARISON RESULTS ===")
    print(f"    {'Method':<18} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Loss':<12}")
    print("    " + "-" * 64)
    for name, result in reg_results.items():
        print(f"    {name:<18} {result['train_acc']:<12.2f} {result['test_acc']:<12.2f} {result['gap']:<10.2f} {result['final_loss']:<12.6f}")

    # ============ Step 12: Optimizer Comparison ============
    print("\n[12] Comparing different optimizers (GD vs Momentum vs RMSprop vs Adam)...")
    opt_results = plot_optimizer_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500)

    print("\n    === OPTIMIZER COMPARISON RESULTS ===")
    print(f"    {'Optimizer':<20} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("    " + "-" * 56)
    for name, result in opt_results.items():
        print(f"    {name:<20} {result['train_acc']:<12.2f} {result['test_acc']:<12.2f} {result['final_loss']:<12.6f}")

    # ============ Step 13: Mini-Batch Comparison ============
    print("\n[13] Comparing different mini-batch sizes...")
    batch_results = plot_mini_batch_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=500)

    print("\n    === MINI-BATCH SIZE COMPARISON RESULTS ===")
    print(f"    {'Batch Size':<20} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("    " + "-" * 56)
    for name, result in batch_results.items():
        print(f"    {name:<20} {result['train_acc']:<12.2f} {result['test_acc']:<12.2f} {result['final_loss']:<12.6f}")

    # ============ Step 14: Learning Rate Comparison ============
    print("\n[14] Comparing different learning rates...")
    learning_rates = [0.001, 0.0075, 0.01, 0.05]
    lr_results = plot_learning_rate_comparison(train_x, train_y, layer_dims, learning_rates, epochs=2500)

    print("\n    === LEARNING RATE COMPARISON RESULTS ===")
    for lr, result in lr_results.items():
        test_acc = result['model'].accuracy(test_x, test_y)
        print(f"    lr = {lr:6.4f} | Final Loss: {result['final_loss']:.6f} | Test Accuracy: {test_acc:.2f}%")

    # ============ Step 15: Learning Rate Decay Comparison ============
    print("\n[15] Comparing different learning rate decay strategies...")
    lr_decay_results = plot_learning_rate_decay_comparison(train_x, train_y, test_x, test_y, layer_dims, epochs=2500)

    print("\n    === LEARNING RATE DECAY COMPARISON RESULTS ===")
    print(f"    {'Decay Strategy':<25} {'Train Acc':<12} {'Test Acc':<12} {'Final LR':<12} {'Loss':<12}")
    print("    " + "-" * 73)
    for name, result in lr_decay_results.items():
        print(f"    {name:<25} {result['train_acc']:<12.2f} {result['test_acc']:<12.2f} {result['final_lr']:<12.6f} {result['final_loss']:<12.6f}")

    # ============ Step 16: Sample Predictions (if real data) ============
    if has_real_data and test_x_orig is not None:
        print("\n[16] Plotting sample predictions...")
        plot_sample_predictions(test_x_orig, test_y, test_predictions, classes)

    # ============ Summary ============
    print("\n" + "=" * 60)
    print("    TRAINING COMPLETE!")
    print("=" * 60)
    print("\n    Generated Graphs:")
    print("    - cost_curve.png")
    print("    - confusion_matrix.png")
    print("    - metrics_comparison.png")
    if has_real_data:
        print("    - normalization_comparison.png")
    print("    - initialization_comparison.png")
    print("    - regularization_comparison.png")
    print("    - regularization_accuracy.png")
    print("    - optimizer_comparison.png")
    print("    - mini_batch_comparison.png")
    print("    - learning_rate_comparison.png")
    print("    - learning_rate_decay_comparison.png")
    if has_real_data:
        print("    - sample_predictions.png")

    print("\n    Key Findings:")
    print("    - Normalization speeds up convergence and improves stability")
    print("    - Z-score standardization is generally recommended")
    print("    - He initialization works best for ReLU networks")
    print("    - L2 regularization and dropout reduce overfitting")
    print("    - Combining L2 + dropout often gives best generalization")
    print("    - Adam optimizer generally performs best")
    print("    - Mini-batch sizes of 32-64 offer good speed/stability balance")
    print("    - Learning rate decay helps fine-tune convergence in later epochs")
    print("    - Scheduled decay (step decay) gives more control over when LR reduces")
    print("\n" + "=" * 60)
