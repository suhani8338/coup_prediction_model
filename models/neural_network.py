"""Feedforward Neural Network with Backpropagation for binary classification"""

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, optimizer='adam', 
                 l2_lambda=0.0, dropout_rate=0.0, random_seed=42):
        """Initialize neural network with specified architecture and hyperparameters."""
        np.random.seed(random_seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.optimizer = optimizer.lower()
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases using He initialization
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization for ReLU activation
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Initialize optimizer-specific parameters
        self._initialize_optimizer()
    
    def _initialize_optimizer(self):
        """Initialize optimizer-specific variables."""
        if self.optimizer == 'adam':
            # Adam: Moving averages of gradients and squared gradients
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.t = 0  # Time step for Adam
        
        elif self.optimizer == 'rmsprop':
            # RMSprop: Moving average of squared gradients
            self.cache_weights = [np.zeros_like(w) for w in self.weights]
            self.cache_biases = [np.zeros_like(b) for b in self.biases]
            self.decay_rate = 0.9
            self.epsilon = 1e-8
    
    def sigmoid(self, z):
        """Sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """Sigmoid derivative."""
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU activation."""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """ReLU derivative."""
        return (z > 0).astype(float)
    
    def tanh(self, z):
        """Tanh activation."""
        return np.tanh(z)
    
    def tanh_derivative(self, a):
        """Tanh derivative."""
        return 1 - a**2
    
    def forward(self, X, training=True):
        """Forward propagation. Applies dropout if training=True."""
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Use ReLU for hidden layers, sigmoid for output layer
            if i < self.num_layers - 2:
                a = self.relu(z)
                
                # Apply dropout to hidden layers during training
                if training and self.dropout_rate > 0:
                    dropout_mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    a = a * dropout_mask
                    self.dropout_masks.append(dropout_mask)
                else:
                    self.dropout_masks.append(None)
            else:
                a = self.sigmoid(z)
                self.dropout_masks.append(None)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss with optional L2 regularization."""
        m = y_true.shape[0]
        y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
        
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        
        if self.l2_lambda > 0:
            l2_loss = (self.l2_lambda / (2 * m)) * sum(np.sum(w**2) for w in self.weights)
            return bce_loss + l2_loss
        
        return bce_loss
    
    def backward(self, X, y):
        """Backward propagation to compute gradients."""
        m = X.shape[0]
        delta = self.activations[-1] - y
        
        weight_gradients = []
        bias_gradients = []
        
        for i in range(self.num_layers - 2, -1, -1):
            dW = np.dot(self.activations[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            if self.l2_lambda > 0:
                dW += (self.l2_lambda / m) * self.weights[i]
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
                
                if self.dropout_masks[i-1] is not None:
                    delta = delta * self.dropout_masks[i-1]
        
        return weight_gradients, bias_gradients
    
    def update_weights(self, weight_gradients, bias_gradients):
        """Update weights and biases using the specified optimizer."""
        if self.optimizer == 'sgd':
            self._update_sgd(weight_gradients, bias_gradients)
        elif self.optimizer == 'adam':
            self._update_adam(weight_gradients, bias_gradients)
        elif self.optimizer == 'rmsprop':
            self._update_rmsprop(weight_gradients, bias_gradients)
    
    def _update_sgd(self, weight_gradients, bias_gradients):
        """SGD update."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def _update_adam(self, weight_gradients, bias_gradients):
        """Adam optimizer update."""
        self.t += 1
        
        for i in range(len(self.weights)):
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_gradients[i]**2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_gradients[i]**2)
            
            m_w_corrected = self.m_weights[i] / (1 - self.beta1**self.t)
            m_b_corrected = self.m_biases[i] / (1 - self.beta1**self.t)
            v_w_corrected = self.v_weights[i] / (1 - self.beta2**self.t)
            v_b_corrected = self.v_biases[i] / (1 - self.beta2**self.t)
            
            self.weights[i] -= self.learning_rate * m_w_corrected / (np.sqrt(v_w_corrected) + self.epsilon)
            self.biases[i] -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + self.epsilon)
    
    def _update_rmsprop(self, weight_gradients, bias_gradients):
        """RMSprop optimizer update."""
        for i in range(len(self.weights)):
            self.cache_weights[i] = self.decay_rate * self.cache_weights[i] + (1 - self.decay_rate) * (weight_gradients[i]**2)
            self.cache_biases[i] = self.decay_rate * self.cache_biases[i] + (1 - self.decay_rate) * (bias_gradients[i]**2)
            
            self.weights[i] -= self.learning_rate * weight_gradients[i] / (np.sqrt(self.cache_weights[i]) + self.epsilon)
            self.biases[i] -= self.learning_rate * bias_gradients[i] / (np.sqrt(self.cache_biases[i]) + self.epsilon)
    
    def fit(self, X, y, epochs=1000, batch_size=32, verbose=True, 
            X_val=None, y_val=None, early_stopping_patience=None, lr_decay=None):
        """Train the neural network with optional validation and early stopping."""
        self.loss_history = []
        self.val_loss_history = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(epochs):
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch, training=True)
                weight_gradients, bias_gradients = self.backward(X_batch, y_batch)
                
                self.update_weights(weight_gradients, bias_gradients)
            
            y_pred_all = self.forward(X, training=False)
            loss = self.compute_loss(y, y_pred_all)
            self.loss_history.append(loss)
            
            if X_val is not None and y_val is not None:
                y_val_pred = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, y_val_pred)
                self.val_loss_history.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    patience_counter += 1
                
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                        print(f"Best validation loss: {best_val_loss:.4f}")
                    self.weights = best_weights
                    self.biases = best_biases
                    break
                
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
            
            if lr_decay is not None and lr_decay < 1.0:
                self.learning_rate = self.initial_learning_rate * (lr_decay ** epoch)
        
        return self
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions using the specified threshold."""
        y_pred = self.forward(X, training=False)
        return (y_pred >= threshold).astype(int)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.forward(X, training=False)
