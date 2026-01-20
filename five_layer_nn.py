import numpy as np

class FiveLayerNN:
    """
    5-Layer Neural Network with configurable layer sizes.

    Architecture:
    Input (X) -> Layer1 -> Layer2 -> Layer3 -> Layer4 -> Layer5 (Output)
    """

    def __init__(self, layer_dims, learning_rate=0.01, initialization='he',
                 lambd=0.0, keep_prob=1.0, optimizer='gd', beta=0.0,
                 beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_rate=0.0, time_interval=1000):
        """
        Initialize the neural network.

        Args:
            layer_dims: list of 6 integers [n_x, n1, n2, n3, n4, n5]
                        where n_x is input size, n5 is output size
            learning_rate: learning rate for gradient descent
            initialization: weight initialization method ('zeros', 'random', 'xavier', 'he')
            lambd: L2 regularization hyperparameter (0 = no regularization)
            keep_prob: dropout keep probability (1.0 = no dropout)
            optimizer: optimization algorithm ('gd', 'momentum', 'rmsprop', 'adam')
            beta: momentum hyperparameter for 'momentum' optimizer (typical: 0.9)
            beta1: exponential decay rate for first moment (Adam), default 0.9
            beta2: exponential decay rate for second moment (Adam/RMSprop), default 0.999
            epsilon: small constant for numerical stability (Adam/RMSprop), default 1e-8
            decay_rate: learning rate decay rate (0 = no decay)
            time_interval: number of epochs between learning rate updates (for scheduled decay)
        """
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.learning_rate0 = learning_rate  # Store initial learning rate for decay
        self.initialization = initialization
        self.lambd = lambd
        self.keep_prob = keep_prob
        self.optimizer = optimizer.lower()
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.time_interval = time_interval
        self.t = 0  # Adam iteration counter for bias correction
        self.parameters = {}
        self.cache = {}
        self.gradients = {}
        self.dropout_masks = {}
        self.velocity = {}  # First moment (momentum/Adam)
        self.squared = {}   # Second moment (RMSprop/Adam)

        self._initialize_parameters()
        self._initialize_optimizer()

    # ==================== Normalization Methods ====================

    @staticmethod
    def normalize_minmax(X, X_min=None, X_max=None):
        """
        Min-Max normalization: scales features to range [0, 1].

        Formula: X_norm = (X - X_min) / (X_max - X_min)

        Args:
            X: Input data, shape (n_features, m_examples)
            X_min: Minimum values per feature. If None, computed from X.
            X_max: Maximum values per feature. If None, computed from X.

        Returns:
            X_norm: Normalized data
            X_min: Minimum values (save for test set normalization)
            X_max: Maximum values (save for test set normalization)
        """
        if X_min is None:
            X_min = np.min(X, axis=1, keepdims=True)
        if X_max is None:
            X_max = np.max(X, axis=1, keepdims=True)

        # Avoid division by zero
        range_vals = X_max - X_min
        range_vals[range_vals == 0] = 1

        X_norm = (X - X_min) / range_vals
        return X_norm, X_min, X_max

    @staticmethod
    def normalize_zscore(X, mean=None, std=None):
        """
        Z-score standardization: transforms to mean=0, std=1.

        Formula: X_norm = (X - mean) / std

        Args:
            X: Input data, shape (n_features, m_examples)
            mean: Mean per feature. If None, computed from X.
            std: Std deviation per feature. If None, computed from X.

        Returns:
            X_norm: Normalized data
            mean: Mean values (save for test set normalization)
            std: Std values (save for test set normalization)
        """
        if mean is None:
            mean = np.mean(X, axis=1, keepdims=True)
        if std is None:
            std = np.std(X, axis=1, keepdims=True)

        # Avoid division by zero
        std[std == 0] = 1

        X_norm = (X - mean) / std
        return X_norm, mean, std

    @staticmethod
    def normalize_mean(X, mean=None):
        """
        Mean normalization: centers data around zero.

        Formula: X_norm = X - mean

        Args:
            X: Input data, shape (n_features, m_examples)
            mean: Mean per feature. If None, computed from X.

        Returns:
            X_norm: Normalized data
            mean: Mean values (save for test set normalization)
        """
        if mean is None:
            mean = np.mean(X, axis=1, keepdims=True)

        X_norm = X - mean
        return X_norm, mean

    @staticmethod
    def normalize_l2(X):
        """
        L2 normalization: scales each sample to unit norm.

        Formula: X_norm = X / ||X||_2

        Args:
            X: Input data, shape (n_features, m_examples)

        Returns:
            X_norm: Normalized data (each column has L2 norm = 1)
        """
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        X_norm = X / norms
        return X_norm

    # ==================== Initialization Methods ====================

    def _initialize_parameters(self):
        """Initialize W and b using the specified initialization method."""
        np.random.seed(42)

        for l in range(1, 6):
            if self.initialization == 'zeros':
                # Zero initialization (bad - causes symmetry problem)
                self.parameters[f'W{l}'] = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))

            elif self.initialization == 'random':
                # Random initialization with large values (can cause vanishing/exploding gradients)
                self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 10

            elif self.initialization == 'xavier':
                # Xavier/Glorot initialization (good for tanh/sigmoid)
                # W = randn * sqrt(1 / n_prev) or sqrt(2 / (n_prev + n_curr))
                self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(1 / self.layer_dims[l-1])

            elif self.initialization == 'he':
                # He initialization (good for ReLU)
                # W = randn * sqrt(2 / n_prev)
                self.parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2 / self.layer_dims[l-1])

            else:
                raise ValueError(f"Unknown initialization: {self.initialization}. Use 'zeros', 'random', 'xavier', or 'he'")

            self.parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))

    def _initialize_optimizer(self):
        """
        Initialize optimizer state variables.

        For Momentum:
            - velocity (v): exponentially weighted average of gradients

        For RMSprop:
            - squared (s): exponentially weighted average of squared gradients

        For Adam:
            - velocity (v): first moment estimate (like momentum)
            - squared (s): second moment estimate (like RMSprop)

        All initialized to zeros.
        """
        for l in range(1, 6):
            # First moment (velocity) - used by momentum and Adam
            self.velocity[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.velocity[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])

            # Second moment (squared gradients) - used by RMSprop and Adam
            self.squared[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.squared[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])

    # ==================== Activation Functions ====================

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    # ==================== Forward Propagation ====================

    def forward_propagation(self, X, training=True):
        """
        Forward pass through all 5 layers.

        Layers 1-4: ReLU activation (with optional dropout)
        Layer 5: Sigmoid activation (binary) or Softmax (multiclass)

        Args:
            X: Input data
            training: If True, apply dropout. If False (inference), no dropout.
        """
        self.cache['A0'] = X
        A = X

        # Layers 1-4: Linear -> ReLU -> Dropout (optional)
        for l in range(1, 5):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            Z = np.dot(W, A) + b
            A = self.relu(Z)

            # Apply dropout during training (not on output layer)
            if training and self.keep_prob < 1.0:
                D = np.random.rand(A.shape[0], A.shape[1])
                D = (D < self.keep_prob).astype(int)
                A = A * D
                A = A / self.keep_prob  # Inverted dropout scaling
                self.dropout_masks[f'D{l}'] = D

            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        # Layer 5: Linear -> Sigmoid (output) - no dropout on output layer
        W5 = self.parameters['W5']
        b5 = self.parameters['b5']

        Z5 = np.dot(W5, A) + b5
        A5 = self.sigmoid(Z5)

        self.cache['Z5'] = Z5
        self.cache['A5'] = A5

        return A5

    # ==================== Backward Propagation ====================

    def backward_propagation(self, Y):
        """
        Backward pass computing dZ, dW, db for all 5 layers.
        Supports L2 regularization and dropout.

        Layer 5 (Output):
            dZ⁵ = A⁵ - Y
            dW⁵ = (1/m) * dZ⁵ · (A⁴)ᵀ + (λ/m) * W⁵
            db⁵ = (1/m) * Σ dZ⁵

        Layers 4-1:
            dAˡ = (Wˡ⁺¹)ᵀ · dZˡ⁺¹
            dAˡ = dAˡ * Dˡ / keep_prob  (if dropout)
            dZˡ = dAˡ * g'ˡ(Zˡ)
            dWˡ = (1/m) * dZˡ · (Aˡ⁻¹)ᵀ + (λ/m) * Wˡ
            dbˡ = (1/m) * Σ dZˡ
        """
        m = Y.shape[1]

        # ============ Layer 5 (Output Layer) ============
        dZ5 = self.cache['A5'] - Y
        dW5 = (1/m) * np.dot(dZ5, self.cache['A4'].T)
        # Add L2 regularization gradient
        if self.lambd > 0:
            dW5 += (self.lambd / m) * self.parameters['W5']
        db5 = (1/m) * np.sum(dZ5, axis=1, keepdims=True)

        self.gradients['dZ5'] = dZ5
        self.gradients['dW5'] = dW5
        self.gradients['db5'] = db5

        # ============ Layer 4 ============
        dA4 = np.dot(self.parameters['W5'].T, dZ5)
        # Apply dropout mask if dropout was used
        if self.keep_prob < 1.0 and f'D4' in self.dropout_masks:
            dA4 = dA4 * self.dropout_masks['D4']
            dA4 = dA4 / self.keep_prob
        dZ4 = dA4 * self.relu_derivative(self.cache['Z4'])
        dW4 = (1/m) * np.dot(dZ4, self.cache['A3'].T)
        if self.lambd > 0:
            dW4 += (self.lambd / m) * self.parameters['W4']
        db4 = (1/m) * np.sum(dZ4, axis=1, keepdims=True)

        self.gradients['dZ4'] = dZ4
        self.gradients['dW4'] = dW4
        self.gradients['db4'] = db4

        # ============ Layer 3 ============
        dA3 = np.dot(self.parameters['W4'].T, dZ4)
        if self.keep_prob < 1.0 and f'D3' in self.dropout_masks:
            dA3 = dA3 * self.dropout_masks['D3']
            dA3 = dA3 / self.keep_prob
        dZ3 = dA3 * self.relu_derivative(self.cache['Z3'])
        dW3 = (1/m) * np.dot(dZ3, self.cache['A2'].T)
        if self.lambd > 0:
            dW3 += (self.lambd / m) * self.parameters['W3']
        db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

        self.gradients['dZ3'] = dZ3
        self.gradients['dW3'] = dW3
        self.gradients['db3'] = db3

        # ============ Layer 2 ============
        dA2 = np.dot(self.parameters['W3'].T, dZ3)
        if self.keep_prob < 1.0 and f'D2' in self.dropout_masks:
            dA2 = dA2 * self.dropout_masks['D2']
            dA2 = dA2 / self.keep_prob
        dZ2 = dA2 * self.relu_derivative(self.cache['Z2'])
        dW2 = (1/m) * np.dot(dZ2, self.cache['A1'].T)
        if self.lambd > 0:
            dW2 += (self.lambd / m) * self.parameters['W2']
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        self.gradients['dZ2'] = dZ2
        self.gradients['dW2'] = dW2
        self.gradients['db2'] = db2

        # ============ Layer 1 ============
        dA1 = np.dot(self.parameters['W2'].T, dZ2)
        if self.keep_prob < 1.0 and f'D1' in self.dropout_masks:
            dA1 = dA1 * self.dropout_masks['D1']
            dA1 = dA1 / self.keep_prob
        dZ1 = dA1 * self.relu_derivative(self.cache['Z1'])
        dW1 = (1/m) * np.dot(dZ1, self.cache['A0'].T)
        if self.lambd > 0:
            dW1 += (self.lambd / m) * self.parameters['W1']
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

        self.gradients['dZ1'] = dZ1
        self.gradients['dW1'] = dW1
        self.gradients['db1'] = db1

    # ==================== Update Parameters ====================

    def update_parameters(self):
        """
        Update W and b using the specified optimizer.

        Supported optimizers:
        1. 'gd' - Standard Gradient Descent:
            W = W - α * dW

        2. 'momentum' - Gradient Descent with Momentum:
            v = β * v + (1-β) * dW
            W = W - α * v
            Smooths gradients, accelerates in consistent directions.

        3. 'rmsprop' - Root Mean Square Propagation:
            s = β₂ * s + (1-β₂) * dW²
            W = W - α * dW / (√s + ε)
            Adapts learning rate per parameter, divides by running average of gradient magnitudes.

        4. 'adam' - Adaptive Moment Estimation:
            v = β₁ * v + (1-β₁) * dW     (first moment - momentum)
            s = β₂ * s + (1-β₂) * dW²    (second moment - RMSprop)
            v_corrected = v / (1 - β₁ᵗ)  (bias correction)
            s_corrected = s / (1 - β₂ᵗ)
            W = W - α * v_corrected / (√s_corrected + ε)
            Combines benefits of momentum and RMSprop with bias correction.
        """
        if self.optimizer == 'adam':
            self.t += 1  # Increment timestep for bias correction

        for l in range(1, 6):
            dW = self.gradients[f'dW{l}']
            db = self.gradients[f'db{l}']

            if self.optimizer == 'gd':
                # Standard gradient descent
                self.parameters[f'W{l}'] -= self.learning_rate * dW
                self.parameters[f'b{l}'] -= self.learning_rate * db

            elif self.optimizer == 'momentum':
                # Momentum: v = β*v + (1-β)*dW, W = W - α*v
                self.velocity[f'dW{l}'] = self.beta * self.velocity[f'dW{l}'] + (1 - self.beta) * dW
                self.velocity[f'db{l}'] = self.beta * self.velocity[f'db{l}'] + (1 - self.beta) * db

                self.parameters[f'W{l}'] -= self.learning_rate * self.velocity[f'dW{l}']
                self.parameters[f'b{l}'] -= self.learning_rate * self.velocity[f'db{l}']

            elif self.optimizer == 'rmsprop':
                # RMSprop: s = β₂*s + (1-β₂)*dW², W = W - α*dW/(√s + ε)
                self.squared[f'dW{l}'] = self.beta2 * self.squared[f'dW{l}'] + (1 - self.beta2) * np.square(dW)
                self.squared[f'db{l}'] = self.beta2 * self.squared[f'db{l}'] + (1 - self.beta2) * np.square(db)

                self.parameters[f'W{l}'] -= self.learning_rate * dW / (np.sqrt(self.squared[f'dW{l}']) + self.epsilon)
                self.parameters[f'b{l}'] -= self.learning_rate * db / (np.sqrt(self.squared[f'db{l}']) + self.epsilon)

            elif self.optimizer == 'adam':
                # Adam: combines momentum and RMSprop with bias correction
                # Update first moment (momentum)
                self.velocity[f'dW{l}'] = self.beta1 * self.velocity[f'dW{l}'] + (1 - self.beta1) * dW
                self.velocity[f'db{l}'] = self.beta1 * self.velocity[f'db{l}'] + (1 - self.beta1) * db

                # Update second moment (RMSprop)
                self.squared[f'dW{l}'] = self.beta2 * self.squared[f'dW{l}'] + (1 - self.beta2) * np.square(dW)
                self.squared[f'db{l}'] = self.beta2 * self.squared[f'db{l}'] + (1 - self.beta2) * np.square(db)

                # Bias correction
                v_dW_corrected = self.velocity[f'dW{l}'] / (1 - self.beta1 ** self.t)
                v_db_corrected = self.velocity[f'db{l}'] / (1 - self.beta1 ** self.t)
                s_dW_corrected = self.squared[f'dW{l}'] / (1 - self.beta2 ** self.t)
                s_db_corrected = self.squared[f'db{l}'] / (1 - self.beta2 ** self.t)

                # Update parameters
                self.parameters[f'W{l}'] -= self.learning_rate * v_dW_corrected / (np.sqrt(s_dW_corrected) + self.epsilon)
                self.parameters[f'b{l}'] -= self.learning_rate * v_db_corrected / (np.sqrt(s_db_corrected) + self.epsilon)

            else:
                raise ValueError(f"Unknown optimizer: {self.optimizer}. Use 'gd', 'momentum', 'rmsprop', or 'adam'")

    # ==================== Learning Rate Decay ====================

    def update_learning_rate(self, epoch_num):
        """
        Update learning rate using inverse time decay formula.

        Formula: learning_rate = learning_rate0 / (1 + decay_rate * epoch_num)

        Args:
            epoch_num: Current epoch number

        Returns:
            learning_rate: Updated learning rate
        """
        if self.decay_rate > 0:
            self.learning_rate = self.learning_rate0 / (1 + self.decay_rate * epoch_num)
        return self.learning_rate

    def schedule_lr_decay(self, epoch_num):
        """
        Update learning rate using scheduled decay (step decay).

        The learning rate is reduced at fixed intervals (time_interval epochs).

        Formula: learning_rate = learning_rate0 / (1 + decay_rate * floor(epoch_num / time_interval))

        Args:
            epoch_num: Current epoch number

        Returns:
            learning_rate: Updated learning rate
        """
        if self.decay_rate > 0:
            self.learning_rate = self.learning_rate0 / (1 + self.decay_rate * np.floor(epoch_num / self.time_interval))
        return self.learning_rate

    # ==================== Loss Function ====================

    def compute_loss(self, Y):
        """Binary cross-entropy loss with optional L2 regularization."""
        m = Y.shape[1]
        A5 = self.cache['A5']

        # Clip to prevent log(0)
        A5 = np.clip(A5, 1e-15, 1 - 1e-15)

        # Cross-entropy loss
        cross_entropy_loss = -(1/m) * np.sum(Y * np.log(A5) + (1 - Y) * np.log(1 - A5))

        # L2 regularization cost
        L2_cost = 0
        if self.lambd > 0:
            for l in range(1, 6):
                L2_cost += np.sum(np.square(self.parameters[f'W{l}']))
            L2_cost = (self.lambd / (2 * m)) * L2_cost

        return cross_entropy_loss + L2_cost

    # ==================== Training ====================

    @staticmethod
    def create_mini_batches(X, Y, mini_batch_size, seed=None):
        """
        Create mini-batches from the training data.

        Args:
            X: Input data, shape (n_x, m)
            Y: Labels, shape (1, m)
            mini_batch_size: Size of each mini-batch
            seed: Random seed for shuffling (optional)

        Returns:
            mini_batches: List of (mini_batch_X, mini_batch_Y) tuples
        """
        if seed is not None:
            np.random.seed(seed)

        m = X.shape[1]
        mini_batches = []

        # Shuffle training data
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]

        # Partition into mini-batches
        num_complete_batches = m // mini_batch_size

        for k in range(num_complete_batches):
            start = k * mini_batch_size
            end = (k + 1) * mini_batch_size
            mini_batch_X = X_shuffled[:, start:end]
            mini_batch_Y = Y_shuffled[:, start:end]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        # Handle the remaining examples (last mini-batch)
        if m % mini_batch_size != 0:
            mini_batch_X = X_shuffled[:, num_complete_batches * mini_batch_size:]
            mini_batch_Y = Y_shuffled[:, num_complete_batches * mini_batch_size:]
            mini_batches.append((mini_batch_X, mini_batch_Y))

        return mini_batches

    def train(self, X, Y, epochs=1000, print_loss=True, mini_batch_size=None, decay_type='scheduled'):
        """
        Train the neural network.

        Args:
            X: Input data, shape (n_x, m)
            Y: Labels, shape (1, m)
            epochs: Number of training epochs
            print_loss: Whether to print loss every 100 epochs
            mini_batch_size: Size of mini-batches. If None, use full batch (batch GD).
                            Common values: 32, 64, 128, 256
            decay_type: Type of learning rate decay ('continuous' or 'scheduled')
                       - 'continuous': Decay every epoch using update_learning_rate()
                       - 'scheduled': Decay at intervals using schedule_lr_decay()

        Returns:
            losses: List of loss values (one per epoch)

        Mini-batch Gradient Descent:
        - mini_batch_size = m: Batch gradient descent (smooth but slow)
        - mini_batch_size = 1: Stochastic gradient descent (noisy but fast)
        - mini_batch_size = 32-256: Mini-batch (good balance)
        """
        losses = []
        learning_rates = []  # Track learning rate over epochs
        m = X.shape[1]

        # Reset Adam timestep counter at start of training
        self.t = 0

        # Reset learning rate to initial value
        self.learning_rate = self.learning_rate0

        # Use full batch if mini_batch_size not specified
        if mini_batch_size is None or mini_batch_size >= m:
            mini_batch_size = m

        for epoch in range(epochs):
            epoch_cost = 0

            # Apply learning rate decay at the start of each epoch
            if self.decay_rate > 0:
                if decay_type == 'continuous':
                    self.update_learning_rate(epoch)
                else:  # 'scheduled'
                    self.schedule_lr_decay(epoch)

            learning_rates.append(self.learning_rate)

            # Create mini-batches (shuffle each epoch for mini-batch GD)
            if mini_batch_size < m:
                mini_batches = self.create_mini_batches(X, Y, mini_batch_size, seed=epoch)
            else:
                mini_batches = [(X, Y)]

            num_batches = len(mini_batches)

            for mini_batch_X, mini_batch_Y in mini_batches:
                # Forward propagation (with dropout if enabled)
                self.forward_propagation(mini_batch_X, training=True)

                # Compute loss for this mini-batch
                batch_cost = self.compute_loss(mini_batch_Y)
                epoch_cost += batch_cost

                # Backward propagation
                self.backward_propagation(mini_batch_Y)

                # Update parameters
                self.update_parameters()

            # Average cost over all mini-batches
            epoch_cost /= num_batches
            losses.append(epoch_cost)

            if print_loss and epoch % 100 == 0:
                lr_info = f" | LR: {self.learning_rate:.6f}" if self.decay_rate > 0 else ""
                print(f"Epoch {epoch:4d} | Loss: {epoch_cost:.6f}{lr_info}")

        # Store learning rate history for analysis
        self.learning_rate_history = learning_rates

        return losses

    # ==================== Prediction ====================

    def predict(self, X):
        """Make predictions (no dropout during inference)."""
        A5 = self.forward_propagation(X, training=False)
        return (A5 > 0.5).astype(int)

    def accuracy(self, X, Y):
        """Compute accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == Y) * 100

    def get_config(self):
        """Return model configuration as a string."""
        config = f"init={self.initialization}"
        if self.lambd > 0:
            config += f", L2={self.lambd}"
        if self.keep_prob < 1.0:
            config += f", dropout={1-self.keep_prob:.1f}"
        if self.optimizer != 'gd':
            config += f", optimizer={self.optimizer}"
            if self.optimizer == 'momentum':
                config += f"(β={self.beta})"
            elif self.optimizer == 'rmsprop':
                config += f"(β₂={self.beta2})"
            elif self.optimizer == 'adam':
                config += f"(β₁={self.beta1},β₂={self.beta2})"
        if self.decay_rate > 0:
            config += f", lr_decay={self.decay_rate}(interval={self.time_interval})"
        return config

    # ==================== Gradient Checking ====================

    def _parameters_to_vector(self):
        """Flatten all parameters into a single vector."""
        params = []
        for l in range(1, 6):
            params.append(self.parameters[f'W{l}'].flatten())
            params.append(self.parameters[f'b{l}'].flatten())
        return np.concatenate(params)

    def _vector_to_parameters(self, theta):
        """Reshape vector back into parameters dictionary."""
        parameters = {}
        idx = 0
        for l in range(1, 6):
            W_shape = (self.layer_dims[l], self.layer_dims[l-1])
            b_shape = (self.layer_dims[l], 1)

            W_size = W_shape[0] * W_shape[1]
            b_size = b_shape[0]

            parameters[f'W{l}'] = theta[idx:idx + W_size].reshape(W_shape)
            idx += W_size
            parameters[f'b{l}'] = theta[idx:idx + b_size].reshape(b_shape)
            idx += b_size

        return parameters

    def _gradients_to_vector(self):
        """Flatten all gradients into a single vector."""
        grads = []
        for l in range(1, 6):
            grads.append(self.gradients[f'dW{l}'].flatten())
            grads.append(self.gradients[f'db{l}'].flatten())
        return np.concatenate(grads)

    def gradient_check(self, X, Y, epsilon=1e-7):
        """
        Perform gradient checking to verify backpropagation.

        Compares analytical gradients (from backprop) with numerical gradients
        (using two-sided finite difference approximation).

        IMPORTANT: Dropout must be disabled during gradient checking because
        random masks would be regenerated on each forward pass, causing
        inconsistent loss values and incorrect numerical gradients.

        Args:
            X: Input data, shape (n_x, m)
            Y: Labels, shape (1, m)
            epsilon: Small perturbation for numerical gradient

        Returns:
            difference: Relative difference between analytical and numerical gradients
                       (should be < 1e-7 if backprop is correct)
        """
        # Save original keep_prob and temporarily disable dropout for gradient checking
        original_keep_prob = self.keep_prob
        self.keep_prob = 1.0  # Disable dropout

        # Compute analytical gradients via backprop (no dropout)
        self.forward_propagation(X, training=False)
        self.backward_propagation(Y)
        analytical_grads = self._gradients_to_vector()

        # Store original parameters
        original_params = self._parameters_to_vector()
        num_parameters = len(original_params)

        # Compute numerical gradients
        numerical_grads = np.zeros(num_parameters)

        for i in range(num_parameters):
            # Compute J(theta + epsilon)
            theta_plus = original_params.copy()
            theta_plus[i] += epsilon
            self.parameters = self._vector_to_parameters(theta_plus)
            self.forward_propagation(X, training=False)
            J_plus = self.compute_loss(Y)

            # Compute J(theta - epsilon)
            theta_minus = original_params.copy()
            theta_minus[i] -= epsilon
            self.parameters = self._vector_to_parameters(theta_minus)
            self.forward_propagation(X, training=False)
            J_minus = self.compute_loss(Y)

            # Two-sided numerical gradient
            numerical_grads[i] = (J_plus - J_minus) / (2 * epsilon)

        # Restore original parameters and keep_prob
        self.parameters = self._vector_to_parameters(original_params)
        self.keep_prob = original_keep_prob  # Restore dropout setting

        # Compute relative difference
        numerator = np.linalg.norm(analytical_grads - numerical_grads)
        denominator = np.linalg.norm(analytical_grads) + np.linalg.norm(numerical_grads)
        difference = numerator / (denominator + 1e-8)

        return difference, analytical_grads, numerical_grads


# ==================== Demo ====================

def compare_initializations(X_train, Y_train, X_test, Y_test, layer_dims, epochs=1500):
    """Compare different initialization methods."""
    print("\n" + "=" * 70)
    print("INITIALIZATION COMPARISON")
    print("=" * 70)

    initializations = ['zeros', 'random', 'xavier', 'he']
    results = {}

    for init in initializations:
        print(f"\n--- Training with {init.upper()} initialization ---")
        np.random.seed(42)

        nn = FiveLayerNN(layer_dims, learning_rate=0.1, initialization=init)
        losses = nn.train(X_train, Y_train, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(X_train, Y_train)
        test_acc = nn.accuracy(X_test, Y_test)

        results[init] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_loss': losses[-1] if losses else float('inf'),
            'losses': losses
        }

        print(f"  Final Loss: {losses[-1]:.6f}" if losses else "  Final Loss: N/A")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")

    print("\n" + "-" * 70)
    print("INITIALIZATION SUMMARY")
    print("-" * 70)
    print(f"{'Method':<12} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    for init, res in results.items():
        print(f"{init:<12} {res['train_acc']:<12.2f} {res['test_acc']:<12.2f} {res['final_loss']:<12.6f}")

    return results


def compare_normalizations(X_train_raw, Y_train, X_test_raw, Y_test, layer_dims, epochs=1500):
    """Compare different normalization methods."""
    print("\n" + "=" * 70)
    print("NORMALIZATION COMPARISON")
    print("=" * 70)

    results = {}

    # 1. No normalization
    print("\n--- Training with NO normalization ---")
    np.random.seed(42)
    nn = FiveLayerNN(layer_dims, learning_rate=0.1, initialization='he')
    losses = nn.train(X_train_raw, Y_train, epochs=epochs, print_loss=False)
    train_acc = nn.accuracy(X_train_raw, Y_train)
    test_acc = nn.accuracy(X_test_raw, Y_test)
    results['None'] = {'train_acc': train_acc, 'test_acc': test_acc, 'final_loss': losses[-1]}
    print(f"  Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    # 2. Min-Max normalization
    print("\n--- Training with MIN-MAX normalization ---")
    X_train_mm, X_min, X_max = FiveLayerNN.normalize_minmax(X_train_raw)
    X_test_mm, _, _ = FiveLayerNN.normalize_minmax(X_test_raw, X_min, X_max)
    np.random.seed(42)
    nn = FiveLayerNN(layer_dims, learning_rate=0.1, initialization='he')
    losses = nn.train(X_train_mm, Y_train, epochs=epochs, print_loss=False)
    train_acc = nn.accuracy(X_train_mm, Y_train)
    test_acc = nn.accuracy(X_test_mm, Y_test)
    results['Min-Max'] = {'train_acc': train_acc, 'test_acc': test_acc, 'final_loss': losses[-1]}
    print(f"  Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    # 3. Z-score standardization
    print("\n--- Training with Z-SCORE standardization ---")
    X_train_zs, mean, std = FiveLayerNN.normalize_zscore(X_train_raw)
    X_test_zs, _, _ = FiveLayerNN.normalize_zscore(X_test_raw, mean, std)
    np.random.seed(42)
    nn = FiveLayerNN(layer_dims, learning_rate=0.1, initialization='he')
    losses = nn.train(X_train_zs, Y_train, epochs=epochs, print_loss=False)
    train_acc = nn.accuracy(X_train_zs, Y_train)
    test_acc = nn.accuracy(X_test_zs, Y_test)
    results['Z-Score'] = {'train_acc': train_acc, 'test_acc': test_acc, 'final_loss': losses[-1]}
    print(f"  Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    # 4. Mean normalization
    print("\n--- Training with MEAN normalization ---")
    X_train_mn, mean = FiveLayerNN.normalize_mean(X_train_raw)
    X_test_mn, _ = FiveLayerNN.normalize_mean(X_test_raw, mean)
    np.random.seed(42)
    nn = FiveLayerNN(layer_dims, learning_rate=0.1, initialization='he')
    losses = nn.train(X_train_mn, Y_train, epochs=epochs, print_loss=False)
    train_acc = nn.accuracy(X_train_mn, Y_train)
    test_acc = nn.accuracy(X_test_mn, Y_test)
    results['Mean'] = {'train_acc': train_acc, 'test_acc': test_acc, 'final_loss': losses[-1]}
    print(f"  Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    # 5. L2 normalization
    print("\n--- Training with L2 normalization ---")
    X_train_l2 = FiveLayerNN.normalize_l2(X_train_raw)
    X_test_l2 = FiveLayerNN.normalize_l2(X_test_raw)
    np.random.seed(42)
    nn = FiveLayerNN(layer_dims, learning_rate=0.1, initialization='he')
    losses = nn.train(X_train_l2, Y_train, epochs=epochs, print_loss=False)
    train_acc = nn.accuracy(X_train_l2, Y_train)
    test_acc = nn.accuracy(X_test_l2, Y_test)
    results['L2'] = {'train_acc': train_acc, 'test_acc': test_acc, 'final_loss': losses[-1]}
    print(f"  Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    print("\n" + "-" * 70)
    print("NORMALIZATION SUMMARY")
    print("-" * 70)
    print(f"{'Method':<12} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    for method, res in results.items():
        print(f"{method:<12} {res['train_acc']:<12.2f} {res['test_acc']:<12.2f} {res['final_loss']:<12.6f}")

    return results


def compare_regularizations(X_train, Y_train, X_test, Y_test, layer_dims, epochs=1500):
    """Compare different regularization methods."""
    print("\n" + "=" * 70)
    print("REGULARIZATION COMPARISON")
    print("=" * 70)

    configs = [
        {'name': 'No Regularization', 'lambd': 0.0, 'keep_prob': 1.0},
        {'name': 'L2 (λ=0.1)', 'lambd': 0.1, 'keep_prob': 1.0},
        {'name': 'L2 (λ=0.5)', 'lambd': 0.5, 'keep_prob': 1.0},
        {'name': 'L2 (λ=1.0)', 'lambd': 1.0, 'keep_prob': 1.0},
        {'name': 'Dropout (0.2)', 'lambd': 0.0, 'keep_prob': 0.8},
        {'name': 'Dropout (0.4)', 'lambd': 0.0, 'keep_prob': 0.6},
        {'name': 'L2 + Dropout', 'lambd': 0.3, 'keep_prob': 0.8},
    ]

    results = {}

    for cfg in configs:
        print(f"\n--- Training with {cfg['name']} ---")
        np.random.seed(42)

        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.1,
            initialization='he',
            lambd=cfg['lambd'],
            keep_prob=cfg['keep_prob']
        )
        losses = nn.train(X_train, Y_train, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(X_train, Y_train)
        test_acc = nn.accuracy(X_test, Y_test)

        results[cfg['name']] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_loss': losses[-1],
            'losses': losses,
            'config': cfg
        }

        print(f"  Final Loss: {losses[-1]:.6f}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Overfitting Gap: {train_acc - test_acc:.2f}%")

    print("\n" + "-" * 70)
    print("REGULARIZATION SUMMARY")
    print("-" * 70)
    print(f"{'Method':<20} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10} {'Loss':<12}")
    print("-" * 70)
    for name, res in results.items():
        gap = res['train_acc'] - res['test_acc']
        print(f"{name:<20} {res['train_acc']:<12.2f} {res['test_acc']:<12.2f} {gap:<10.2f} {res['final_loss']:<12.6f}")

    return results


def compare_optimizers(X_train, Y_train, X_test, Y_test, layer_dims, epochs=1500):
    """Compare different optimization algorithms."""
    print("\n" + "=" * 70)
    print("OPTIMIZER COMPARISON")
    print("=" * 70)

    configs = [
        {'name': 'Gradient Descent', 'optimizer': 'gd', 'lr': 0.1},
        {'name': 'Momentum (β=0.9)', 'optimizer': 'momentum', 'beta': 0.9, 'lr': 0.1},
        {'name': 'RMSprop', 'optimizer': 'rmsprop', 'beta2': 0.999, 'lr': 0.01},
        {'name': 'Adam', 'optimizer': 'adam', 'beta1': 0.9, 'beta2': 0.999, 'lr': 0.01},
    ]

    results = {}

    for cfg in configs:
        print(f"\n--- Training with {cfg['name']} ---")
        np.random.seed(42)

        nn = FiveLayerNN(
            layer_dims,
            learning_rate=cfg.get('lr', 0.01),
            initialization='he',
            optimizer=cfg['optimizer'],
            beta=cfg.get('beta', 0.9),
            beta1=cfg.get('beta1', 0.9),
            beta2=cfg.get('beta2', 0.999)
        )
        losses = nn.train(X_train, Y_train, epochs=epochs, print_loss=False)

        train_acc = nn.accuracy(X_train, Y_train)
        test_acc = nn.accuracy(X_test, Y_test)

        results[cfg['name']] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_loss': losses[-1],
            'losses': losses,
            'config': cfg
        }

        print(f"  Final Loss: {losses[-1]:.6f}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")

    print("\n" + "-" * 70)
    print("OPTIMIZER SUMMARY")
    print("-" * 70)
    print(f"{'Optimizer':<20} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<20} {res['train_acc']:<12.2f} {res['test_acc']:<12.2f} {res['final_loss']:<12.6f}")

    return results


def compare_mini_batch_sizes(X_train, Y_train, X_test, Y_test, layer_dims, epochs=500):
    """Compare different mini-batch sizes."""
    print("\n" + "=" * 70)
    print("MINI-BATCH SIZE COMPARISON")
    print("=" * 70)

    m = X_train.shape[1]
    batch_sizes = [1, 32, 64, 128, m]  # SGD, mini-batch sizes, full batch
    batch_names = ['SGD (1)', 'Mini-batch (32)', 'Mini-batch (64)', 'Mini-batch (128)', f'Batch GD ({m})']

    results = {}

    for batch_size, name in zip(batch_sizes, batch_names):
        print(f"\n--- Training with {name} ---")
        np.random.seed(42)

        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.01,
            initialization='he',
            optimizer='adam'
        )
        losses = nn.train(X_train, Y_train, epochs=epochs, print_loss=False,
                         mini_batch_size=batch_size)

        train_acc = nn.accuracy(X_train, Y_train)
        test_acc = nn.accuracy(X_test, Y_test)

        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_loss': losses[-1],
            'losses': losses,
            'batch_size': batch_size
        }

        print(f"  Final Loss: {losses[-1]:.6f}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")

    print("\n" + "-" * 70)
    print("MINI-BATCH SIZE SUMMARY")
    print("-" * 70)
    print(f"{'Batch Size':<20} {'Train Acc':<12} {'Test Acc':<12} {'Final Loss':<12}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<20} {res['train_acc']:<12.2f} {res['test_acc']:<12.2f} {res['final_loss']:<12.6f}")

    return results


def compare_learning_rate_decay(X_train, Y_train, X_test, Y_test, layer_dims, epochs=2500):
    """Compare different learning rate decay strategies."""
    print("\n" + "=" * 70)
    print("LEARNING RATE DECAY COMPARISON")
    print("=" * 70)

    configs = [
        {'name': 'No Decay', 'decay_rate': 0.0, 'time_interval': 1000, 'decay_type': 'scheduled'},
        {'name': 'Continuous (rate=0.01)', 'decay_rate': 0.01, 'time_interval': 1000, 'decay_type': 'continuous'},
        {'name': 'Continuous (rate=0.1)', 'decay_rate': 0.1, 'time_interval': 1000, 'decay_type': 'continuous'},
        {'name': 'Scheduled (rate=1, int=500)', 'decay_rate': 1.0, 'time_interval': 500, 'decay_type': 'scheduled'},
        {'name': 'Scheduled (rate=1, int=1000)', 'decay_rate': 1.0, 'time_interval': 1000, 'decay_type': 'scheduled'},
        {'name': 'Scheduled (rate=0.5, int=500)', 'decay_rate': 0.5, 'time_interval': 500, 'decay_type': 'scheduled'},
    ]

    results = {}

    for cfg in configs:
        print(f"\n--- Training with {cfg['name']} ---")
        np.random.seed(42)

        nn = FiveLayerNN(
            layer_dims,
            learning_rate=0.1,
            initialization='he',
            optimizer='gd',
            decay_rate=cfg['decay_rate'],
            time_interval=cfg['time_interval']
        )
        losses = nn.train(X_train, Y_train, epochs=epochs, print_loss=False,
                         decay_type=cfg['decay_type'])

        train_acc = nn.accuracy(X_train, Y_train)
        test_acc = nn.accuracy(X_test, Y_test)

        # Get final learning rate
        final_lr = nn.learning_rate

        results[cfg['name']] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'final_loss': losses[-1],
            'final_lr': final_lr,
            'losses': losses,
            'lr_history': nn.learning_rate_history if hasattr(nn, 'learning_rate_history') else [],
            'config': cfg
        }

        print(f"  Final Loss: {losses[-1]:.6f}")
        print(f"  Final LR: {final_lr:.6f}")
        print(f"  Train Accuracy: {train_acc:.2f}%")
        print(f"  Test Accuracy: {test_acc:.2f}%")

    print("\n" + "-" * 70)
    print("LEARNING RATE DECAY SUMMARY")
    print("-" * 70)
    print(f"{'Decay Strategy':<30} {'Train Acc':<12} {'Test Acc':<12} {'Final LR':<12} {'Loss':<12}")
    print("-" * 70)
    for name, res in results.items():
        print(f"{name:<30} {res['train_acc']:<12.2f} {res['test_acc']:<12.2f} {res['final_lr']:<12.6f} {res['final_loss']:<12.6f}")

    return results


if __name__ == "__main__":
    # Generate sample data (XOR-like problem with noise for overfitting demo)
    np.random.seed(1)
    m_train = 300  # Small training set to show overfitting
    m_test = 200

    # Training data
    X_train = np.random.randn(2, m_train)
    Y_train = ((X_train[0, :] * X_train[1, :]) > 0).astype(int).reshape(1, m_train)
    # Add some noise
    noise_idx = np.random.choice(m_train, size=int(m_train * 0.05), replace=False)
    Y_train[0, noise_idx] = 1 - Y_train[0, noise_idx]

    # Test data (clean)
    X_test = np.random.randn(2, m_test)
    Y_test = ((X_test[0, :] * X_test[1, :]) > 0).astype(int).reshape(1, m_test)

    # Network architecture
    layer_dims = [2, 20, 15, 10, 5, 1]  # Larger network to show overfitting

    print("=" * 70)
    print("5-LAYER NEURAL NETWORK")
    print("Initialization & Regularization Comparison")
    print("=" * 70)
    print(f"Architecture: {layer_dims}")
    print(f"Training samples: {m_train}")
    print(f"Test samples: {m_test}")

    # ============ Compare Normalizations ============
    norm_results = compare_normalizations(X_train, Y_train, X_test, Y_test, layer_dims)

    # ============ Compare Initializations ============
    init_results = compare_initializations(X_train, Y_train, X_test, Y_test, layer_dims)

    # ============ Compare Regularizations ============
    reg_results = compare_regularizations(X_train, Y_train, X_test, Y_test, layer_dims)

    # ============ Compare Optimizers ============
    opt_results = compare_optimizers(X_train, Y_train, X_test, Y_test, layer_dims)

    # ============ Compare Mini-Batch Sizes ============
    batch_results = compare_mini_batch_sizes(X_train, Y_train, X_test, Y_test, layer_dims)

    # ============ Compare Learning Rate Decay ============
    lr_decay_results = compare_learning_rate_decay(X_train, Y_train, X_test, Y_test, layer_dims)

    # ============ Gradient Checking ============
    print("\n" + "=" * 70)
    print("GRADIENT CHECKING")
    print("=" * 70)

    # Create a network for gradient checking
    # Note: Use enough neurons per layer to avoid "dead ReLU" problem
    # where all activations become zero (making gradients zero too)
    grad_check_dims = [2, 10, 8, 6, 4, 1]
    np.random.seed(1)
    grad_check_nn = FiveLayerNN(grad_check_dims, learning_rate=0.1, initialization='he')

    # Use small subset of data
    X_small = X_train[:, :5]
    Y_small = Y_train[:, :5]

    diff, analytical, numerical = grad_check_nn.gradient_check(X_small, Y_small)

    print(f"Relative difference: {diff:.2e}")
    if diff < 1e-7:
        print("Gradient check PASSED! Backpropagation is correct.")
    elif diff < 1e-5:
        print("Gradient check WARNING: Small discrepancy detected.")
    else:
        print("Gradient check FAILED! There may be a bug in backpropagation.")

    # ============ Final Summary ============
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    NORMALIZATION:
    - Min-Max: Scales to [0, 1] range - good for bounded features
    - Z-Score: Mean=0, Std=1 - most common, handles outliers
    - Mean: Centers around zero - simple but effective
    - L2: Unit norm per sample - for direction-based similarity

    INITIALIZATION:
    - Zeros: All neurons learn the same thing (symmetry problem)
    - Random (large): Can cause exploding/vanishing gradients
    - Xavier: Good for tanh/sigmoid activations
    - He: Best for ReLU activations (used in this network)

    REGULARIZATION:
    - L2 (Weight Decay): Penalizes large weights, reduces overfitting
    - Dropout: Randomly drops neurons, prevents co-adaptation
    - Combined: Often gives best results

    OPTIMIZATION ALGORITHMS:
    1. Gradient Descent (GD):
       - W = W - α*dW
       - Simple but can be slow

    2. Momentum:
       - v = β*v + (1-β)*dW, W = W - α*v
       - Accelerates in consistent directions, dampens oscillations
       - β=0.9 is typical (~10 gradients averaged)

    3. RMSprop:
       - s = β₂*s + (1-β₂)*dW², W = W - α*dW/√(s+ε)
       - Adapts learning rate per parameter
       - Good for non-stationary problems

    4. Adam (Recommended):
       - Combines Momentum + RMSprop with bias correction
       - v = β₁*v + (1-β₁)*dW (first moment)
       - s = β₂*s + (1-β₂)*dW² (second moment)
       - Works well in most cases, less sensitive to hyperparameters

    MINI-BATCH GRADIENT DESCENT:
    - Batch GD (batch=m): Smooth but slow, memory intensive
    - SGD (batch=1): Noisy but fast, can escape local minima
    - Mini-batch (32-256): Best of both worlds (recommended)

    LEARNING RATE DECAY:
    - Helps fine-tune convergence in later training stages
    - Two types:
      1. Continuous decay: α = α₀ / (1 + decay_rate * epoch)
         - Smoothly decreases every epoch
      2. Scheduled decay: α = α₀ / (1 + decay_rate * floor(epoch / interval))
         - Steps down at fixed intervals (e.g., every 500 or 1000 epochs)
    - Benefits:
      - Large steps early for fast initial progress
      - Small steps later for fine-tuning near minimum
    - Typical values: decay_rate=0.01-1.0, interval=500-1000

    SIGNS OF OVERFITTING:
    - High training accuracy, low test accuracy
    - Large gap between train and test performance

    RECOMMENDATIONS:
    - Always normalize input data (Z-score is most common)
    - Use He initialization for ReLU networks
    - Use Adam optimizer with mini-batches (64 or 128)
    - Start with small L2 (λ=0.01-0.1) or dropout (keep_prob=0.8-0.9)
    - Increase regularization if overfitting persists
    """)
    print("=" * 70)
