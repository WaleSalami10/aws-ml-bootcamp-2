# Neural Network Troubleshooting Guide

## 🚨 Quick Diagnostic Checklist

When your neural network isn't working as expected, follow this systematic approach:

### 1. First 5 Minutes - Basic Sanity Checks
```python
def quick_sanity_check(model, X_train, Y_train):
    """Run these checks before diving deep into debugging"""
    
    # Check 1: Data shapes and types
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_train dtype: {X_train.dtype}")
    print(f"Y_train dtype: {Y_train.dtype}")
    
    # Check 2: Data ranges
    print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"Y_train unique values: {np.unique(Y_train)}")
    
    # Check 3: No NaN or Inf values
    print(f"X_train has NaN: {np.isnan(X_train).any()}")
    print(f"X_train has Inf: {np.isinf(X_train).any()}")
    
    # Check 4: Class distribution
    if len(np.unique(Y_train)) == 2:  # Binary classification
        class_counts = np.bincount(Y_train.flatten().astype(int))
        print(f"Class distribution: {class_counts}")
        print(f"Class balance ratio: {class_counts.min()/class_counts.max():.3f}")
    
    # Check 5: Model can overfit small batch
    X_small = X_train[:, :5]  # First 5 samples
    Y_small = Y_train[:, :5]
    
    # Train for many iterations on tiny dataset
    model_test = type(model)(model.layer_dims, learning_rate=0.01)
    losses = model_test.train(X_small, Y_small, epochs=1000, print_loss=False)
    
    final_acc = model_test.accuracy(X_small, Y_small)
    print(f"Overfitting test - Final accuracy on 5 samples: {final_acc:.1f}%")
    
    if final_acc < 95:
        print("⚠️  WARNING: Model cannot overfit tiny dataset - check implementation!")
    else:
        print("✅ Model can overfit - implementation likely correct")
```

## 🔍 Problem Categories and Solutions

### Category 1: Loss Not Decreasing

#### Symptom: Loss stays constant or increases
```python
def diagnose_no_learning():
    """
    Possible causes and solutions:
    """
    
    # Cause 1: Learning rate too small
    # Solution: Increase learning rate by 10x
    learning_rates_to_try = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    # Cause 2: Learning rate too large  
    # Symptom: Loss explodes or oscillates wildly
    # Solution: Decrease learning rate by 10x
    
    # Cause 3: Wrong gradient computation
    # Solution: Implement gradient checking
    def gradient_check_debug(model, X, Y):
        diff = model.gradient_check(X, Y)
        if diff > 1e-5:
            print(f"❌ Gradient check failed: {diff:.2e}")
            print("Check backpropagation implementation")
        else:
            print(f"✅ Gradient check passed: {diff:.2e}")
    
    # Cause 4: Vanishing gradients
    # Solution: Check activation distributions
    def check_activations(model, X):
        model.forward_propagation(X, training=False)
        for l in range(1, 6):
            if f'A{l}' in model.cache:
                A = model.cache[f'A{l}']
                print(f"Layer {l} activation stats:")
                print(f"  Mean: {A.mean():.4f}, Std: {A.std():.4f}")
                print(f"  Min: {A.min():.4f}, Max: {A.max():.4f}")
                
                # Check for dead ReLUs
                if l < 5:  # Hidden layers use ReLU
                    dead_neurons = (A == 0).all(axis=1).sum()
                    total_neurons = A.shape[0]
                    print(f"  Dead ReLUs: {dead_neurons}/{total_neurons} ({100*dead_neurons/total_neurons:.1f}%)")
```

#### Quick Fixes for No Learning:
1. **Try different learning rates**: `[0.001, 0.01, 0.1]`
2. **Check data preprocessing**: Ensure inputs are normalized
3. **Verify loss function**: Make sure it matches your problem type
4. **Test on tiny dataset**: Model should overfit 5-10 samples easily

### Category 2: Exploding Gradients

#### Symptom: Loss becomes NaN or very large numbers
```python
def diagnose_exploding_gradients():
    """
    Solutions for exploding gradients:
    """
    
    # Solution 1: Gradient clipping
    def clip_gradients(gradients, max_norm=5.0):
        total_norm = 0
        for grad in gradients.values():
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            for key in gradients:
                if gradients[key] is not None:
                    gradients[key] *= clip_factor
        
        return total_norm
    
    # Solution 2: Better initialization
    # Use He initialization for ReLU networks
    
    # Solution 3: Lower learning rate
    # Start with 0.001 and increase gradually
    
    # Solution 4: Batch normalization
    # Normalizes inputs to each layer
```

### Category 3: Overfitting

#### Symptom: High training accuracy, low test accuracy
```python
def diagnose_overfitting():
    """
    Overfitting solutions in order of preference:
    """
    
    # Solution 1: More data (best solution)
    print("1. Get more training data if possible")
    
    # Solution 2: Regularization
    regularization_configs = [
        {'lambd': 0.01, 'keep_prob': 1.0},   # Light L2
        {'lambd': 0.1, 'keep_prob': 1.0},    # Medium L2
        {'lambd': 0.0, 'keep_prob': 0.8},    # Dropout only
        {'lambd': 0.1, 'keep_prob': 0.8},    # L2 + Dropout
    ]
    
    # Solution 3: Smaller network
    smaller_architectures = [
        [n_x, 10, 5, 1],      # Smaller hidden layers
        [n_x, 15, 1],         # Fewer layers
        [n_x, 1],             # Logistic regression
    ]
    
    # Solution 4: Early stopping
    def early_stopping_example():
        best_val_acc = 0
        patience = 100
        wait = 0
        
        for epoch in range(max_epochs):
            # Train one epoch
            val_acc = model.accuracy(X_val, Y_val)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                wait = 0
                # Save best model
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
```

### Category 4: Underfitting

#### Symptom: Low training and test accuracy
```python
def diagnose_underfitting():
    """
    Underfitting solutions:
    """
    
    # Solution 1: Larger network
    larger_architectures = [
        [n_x, 50, 25, 10, 1],     # More neurons
        [n_x, 20, 15, 10, 5, 3, 1],  # More layers
    ]
    
    # Solution 2: Reduce regularization
    less_regularization = [
        {'lambd': 0.0, 'keep_prob': 1.0},    # No regularization
        {'lambd': 0.001, 'keep_prob': 0.95}, # Very light
    ]
    
    # Solution 3: Better features
    print("Consider feature engineering:")
    print("- Polynomial features")
    print("- Feature interactions") 
    print("- Domain-specific transformations")
    
    # Solution 4: More training
    print("Train for more epochs with learning rate decay")
```

## 🛠️ Debugging Tools and Techniques

### Tool 1: Training Curve Analysis
```python
def analyze_training_curves(losses, train_accs, val_accs):
    """
    What training curves tell you:
    """
    
    # Healthy training curve
    if losses[-1] < losses[0] * 0.1:  # Loss decreased significantly
        print("✅ Loss is decreasing properly")
    else:
        print("⚠️  Loss not decreasing enough")
    
    # Check for overfitting
    if len(train_accs) > 100:  # Need enough data points
        recent_train = np.mean(train_accs[-50:])
        recent_val = np.mean(val_accs[-50:])
        gap = recent_train - recent_val
        
        if gap > 10:  # More than 10% gap
            print(f"⚠️  Overfitting detected: {gap:.1f}% gap")
        elif gap < 2:
            print("✅ Good generalization")
        else:
            print(f"✅ Acceptable gap: {gap:.1f}%")
    
    # Check for convergence
    if len(losses) > 100:
        recent_improvement = losses[-100] - losses[-1]
        if recent_improvement < 0.001:
            print("⚠️  Training may have converged - consider stopping")
```

### Tool 2: Weight and Gradient Monitoring
```python
def monitor_weights_and_gradients(model):
    """
    Monitor training health through weights and gradients
    """
    
    print("Weight Statistics:")
    for l in range(1, 6):
        W = model.parameters[f'W{l}']
        print(f"Layer {l}: mean={W.mean():.4f}, std={W.std():.4f}, "
              f"min={W.min():.4f}, max={W.max():.4f}")
    
    print("\nGradient Statistics:")
    for l in range(1, 6):
        if f'dW{l}' in model.gradients:
            dW = model.gradients[f'dW{l}']
            grad_norm = np.linalg.norm(dW)
            print(f"Layer {l}: gradient norm={grad_norm:.6f}")
            
            # Check for vanishing gradients
            if grad_norm < 1e-8:
                print(f"  ⚠️  Very small gradients in layer {l}")
            
            # Check for exploding gradients  
            if grad_norm > 10:
                print(f"  ⚠️  Large gradients in layer {l}")
```

### Tool 3: Activation Analysis
```python
def analyze_activations(model, X):
    """
    Analyze activation patterns to detect issues
    """
    
    model.forward_propagation(X, training=False)
    
    for l in range(1, 6):
        if f'A{l}' in model.cache:
            A = model.cache[f'A{l}']
            
            print(f"\nLayer {l} Analysis:")
            
            # Basic statistics
            print(f"  Shape: {A.shape}")
            print(f"  Mean: {A.mean():.4f}, Std: {A.std():.4f}")
            
            # Check for saturation (sigmoid/tanh)
            if l == 5:  # Output layer (sigmoid)
                saturated = ((A < 0.01) | (A > 0.99)).mean()
                print(f"  Saturated neurons: {saturated*100:.1f}%")
                if saturated > 0.5:
                    print("  ⚠️  High saturation - consider different initialization")
            
            # Check for dead ReLUs (hidden layers)
            elif l < 5:
                dead_neurons = (A == 0).all(axis=1).mean()
                print(f"  Dead ReLUs: {dead_neurons*100:.1f}%")
                if dead_neurons > 0.5:
                    print("  ⚠️  Many dead ReLUs - consider Leaky ReLU or lower learning rate")
            
            # Check activation distribution
            if A.std() < 0.1:
                print("  ⚠️  Low activation variance - may indicate vanishing gradients")
            elif A.std() > 10:
                print("  ⚠️  High activation variance - may indicate exploding gradients")
```

## 🎯 Systematic Debugging Process

### Step 1: Reproduce the Issue
```python
def reproduce_issue():
    """
    Make the problem reproducible:
    """
    # Fix random seeds
    np.random.seed(42)
    
    # Use same data split
    # Use same hyperparameters
    # Use same initialization
    
    # Document exact steps to reproduce
```

### Step 2: Isolate the Problem
```python
def isolate_problem():
    """
    Test components individually:
    """
    
    # Test 1: Can model overfit tiny dataset?
    X_tiny = X_train[:, :5]
    Y_tiny = Y_train[:, :5]
    # Should reach 100% accuracy
    
    # Test 2: Does gradient checking pass?
    diff = model.gradient_check(X_tiny, Y_tiny)
    # Should be < 1e-7
    
    # Test 3: Are activations reasonable?
    # Check activation statistics
    
    # Test 4: Is data preprocessed correctly?
    # Check input ranges and distributions
```

### Step 3: Systematic Parameter Search
```python
def systematic_search():
    """
    Test one parameter at a time:
    """
    
    # Baseline configuration
    baseline = {
        'learning_rate': 0.01,
        'initialization': 'he',
        'lambd': 0.0,
        'keep_prob': 1.0,
        'optimizer': 'gd'
    }
    
    # Test learning rates
    for lr in [0.001, 0.01, 0.1]:
        config = baseline.copy()
        config['learning_rate'] = lr
        # Train and evaluate
    
    # Test initializations
    for init in ['xavier', 'he']:
        config = baseline.copy()
        config['initialization'] = init
        # Train and evaluate
    
    # Continue for other parameters...
```

## 📊 Performance Benchmarking

### Establishing Baselines
```python
def establish_baselines(X_train, Y_train, X_test, Y_test):
    """
    Create performance baselines to compare against:
    """
    
    # Baseline 1: Random guessing
    random_acc = 50.0  # For balanced binary classification
    print(f"Random baseline: {random_acc:.1f}%")
    
    # Baseline 2: Majority class
    majority_class = np.bincount(Y_train.flatten().astype(int)).argmax()
    majority_acc = (Y_test.flatten() == majority_class).mean() * 100
    print(f"Majority class baseline: {majority_acc:.1f}%")
    
    # Baseline 3: Logistic regression
    from sklearn.linear_model import LogisticRegression
    lr_model = LogisticRegression()
    lr_model.fit(X_train.T, Y_train.flatten())
    lr_acc = lr_model.score(X_test.T, Y_test.flatten()) * 100
    print(f"Sklearn LogisticRegression: {lr_acc:.1f}%")
    
    # Your model should beat these baselines
    return {'random': random_acc, 'majority': majority_acc, 'sklearn_lr': lr_acc}
```

### Performance Comparison Framework
```python
def compare_configurations(X_train, Y_train, X_test, Y_test):
    """
    Systematic comparison of different configurations:
    """
    
    configs = [
        {'name': 'Baseline', 'lr': 0.01, 'init': 'he', 'lambd': 0.0, 'keep_prob': 1.0},
        {'name': 'L2 Reg', 'lr': 0.01, 'init': 'he', 'lambd': 0.1, 'keep_prob': 1.0},
        {'name': 'Dropout', 'lr': 0.01, 'init': 'he', 'lambd': 0.0, 'keep_prob': 0.8},
        {'name': 'Both Reg', 'lr': 0.01, 'init': 'he', 'lambd': 0.1, 'keep_prob': 0.8},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        # Train model with this configuration
        model = FiveLayerNN(
            layer_dims=[X_train.shape[0], 20, 7, 5, 3, 1],
            learning_rate=config['lr'],
            initialization=config['init'],
            lambd=config['lambd'],
            keep_prob=config['keep_prob']
        )
        
        losses = model.train(X_train, Y_train, epochs=1000, print_loss=False)
        
        train_acc = model.accuracy(X_train, Y_train)
        test_acc = model.accuracy(X_test, Y_test)
        
        results[config['name']] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': train_acc - test_acc,
            'final_loss': losses[-1]
        }
        
        print(f"  Train: {train_acc:.1f}%, Test: {test_acc:.1f}%, Gap: {train_acc-test_acc:.1f}%")
    
    return results
```

## 🚀 Advanced Debugging Techniques

### Technique 1: Loss Landscape Visualization
```python
def visualize_loss_landscape(model, X, Y):
    """
    Visualize the loss landscape around current parameters
    """
    
    # Get current parameters
    W1_orig = model.parameters['W1'].copy()
    W2_orig = model.parameters['W2'].copy()
    
    # Create grid around current point
    alpha_range = np.linspace(-1, 1, 21)
    beta_range = np.linspace(-1, 1, 21)
    
    losses = np.zeros((len(alpha_range), len(beta_range)))
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Perturb parameters
            model.parameters['W1'] = W1_orig + alpha * 0.1 * np.random.randn(*W1_orig.shape)
            model.parameters['W2'] = W2_orig + beta * 0.1 * np.random.randn(*W2_orig.shape)
            
            # Compute loss
            model.forward_propagation(X, training=False)
            loss = model.compute_loss(Y)
            losses[i, j] = loss
    
    # Restore original parameters
    model.parameters['W1'] = W1_orig
    model.parameters['W2'] = W2_orig
    
    # Plot landscape
    plt.figure(figsize=(10, 8))
    plt.contour(alpha_range, beta_range, losses.T, levels=20)
    plt.colorbar()
    plt.xlabel('W1 perturbation')
    plt.ylabel('W2 perturbation')
    plt.title('Loss Landscape')
    plt.show()
```

### Technique 2: Learning Rate Range Test
```python
def learning_rate_range_test(model, X_train, Y_train):
    """
    Find optimal learning rate range using cyclical learning rates
    """
    
    min_lr = 1e-7
    max_lr = 10
    num_iterations = 1000
    
    # Exponentially increase learning rate
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iterations)
    losses = []
    
    for i, lr in enumerate(lrs):
        model.learning_rate = lr
        
        # Train one iteration
        model.forward_propagation(X_train, training=True)
        loss = model.compute_loss(Y_train)
        model.backward_propagation(Y_train)
        model.update_parameters()
        
        losses.append(loss)
        
        # Stop if loss explodes
        if loss > losses[0] * 3:
            break
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs[:len(losses)], losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Range Test')
    plt.grid(True)
    plt.show()
    
    # Find optimal range (steepest decrease)
    gradients = np.gradient(losses)
    optimal_idx = np.argmin(gradients)
    optimal_lr = lrs[optimal_idx]
    
    print(f"Suggested learning rate: {optimal_lr:.6f}")
    return optimal_lr
```

## 📋 Debugging Checklist

### Before Training
- [ ] Data shapes are correct
- [ ] No NaN or Inf values in data
- [ ] Input data is normalized
- [ ] Labels are in correct format (0/1 for binary classification)
- [ ] Train/test split is reasonable
- [ ] Model architecture makes sense for problem size

### During Training
- [ ] Loss is decreasing (at least initially)
- [ ] Gradients are not vanishing (> 1e-8) or exploding (< 10)
- [ ] Activations are in reasonable ranges
- [ ] No NaN values appearing in parameters
- [ ] Training and validation curves are reasonable

### After Training
- [ ] Model can overfit small dataset (sanity check)
- [ ] Gradient checking passes (< 1e-7 difference)
- [ ] Performance beats simple baselines
- [ ] Generalization gap is reasonable (< 10%)
- [ ] Results are reproducible with fixed seeds

### Red Flags 🚩
- Loss becomes NaN → Exploding gradients or numerical instability
- Loss doesn't decrease after 100 iterations → Learning rate or implementation issue
- Perfect training accuracy but random test accuracy → Severe overfitting or data leakage
- All activations are 0 → Dead ReLU problem
- Gradients are all very small → Vanishing gradient problem

---

**Remember**: Debugging is a skill that improves with practice. Keep a log of issues you encounter and their solutions - you'll likely see similar patterns in future projects!