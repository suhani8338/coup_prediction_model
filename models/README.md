# Model Improvements Guide

This document explains the improvements made to the neural network and how to use them.

## Improvements Implemented

### 1. **Adam Optimizer (Manual Implementation)**
- **What it is**: Adaptive Moment Estimation - combines momentum and RMSprop
- **How it works**:
  - Tracks moving averages of gradients (first moment)
  - Tracks moving averages of squared gradients (second moment)
  - Applies bias correction for both moments
  - Adapts learning rate per parameter
- **Benefits**: Faster convergence, better handling of sparse gradients, less sensitive to learning rate choice

### 2. **L2 Regularization (Ridge)**
- Adds penalty term to loss: `λ/2m * Σ(w²)`
- Prevents overfitting by keeping weights small
- Parameter: `l2_lambda` (try 0.001, 0.01, 0.1)

### 3. **Dropout Regularization**
- Randomly drops neurons during training
- Forces network to learn robust features
- Parameter: `dropout_rate` (try 0.2-0.5 for hidden layers)

### 4. **Early Stopping**
- Monitors validation loss
- Stops training when no improvement for N epochs
- Restores best weights
- Parameter: `early_stopping_patience`

### 5. **Learning Rate Decay**
- Reduces learning rate over time
- Helps fine-tune weights in later epochs
- Parameter: `lr_decay` (e.g., 0.95 = 5% reduction per epoch)

### 6. **Additional Optimizers**
- **SGD**: Standard gradient descent
- **RMSprop**: Adaptive learning rate based on moving average of squared gradients
- **Adam**: Best default choice

## Usage Examples

### Basic Usage with Adam
```python
from models.neural_network import NeuralNetwork

model = NeuralNetwork(
    layer_sizes=[15, 32, 16, 1],
    learning_rate=0.001,
    optimizer='adam',
    l2_lambda=0.01,
    dropout_rate=0.2
)

model.fit(X_train, y_train, epochs=1000, batch_size=32)
```

### With Early Stopping
```python
model.fit(
    X_train, y_train,
    epochs=2000,
    batch_size=32,
    X_val=X_val,
    y_val=y_val,
    early_stopping_patience=50
)
```

### With Learning Rate Decay
```python
model.fit(
    X_train, y_train,
    epochs=1000,
    batch_size=32,
    lr_decay=0.95  # 5% decay per epoch
)
```

### Comparing Optimizers
```python
# Adam (default - best for most cases)
model_adam = NeuralNetwork(..., optimizer='adam', learning_rate=0.001)

# SGD (slower but sometimes more stable)
model_sgd = NeuralNetwork(..., optimizer='sgd', learning_rate=0.01)

# RMSprop (good for RNNs, between SGD and Adam)
model_rmsprop = NeuralNetwork(..., optimizer='rmsprop', learning_rate=0.001)
```

## Hyperparameter Tuning Recommendations

### For Adam Optimizer:
- **Learning rate**: 0.001 (default), try 0.0001 - 0.01
- **L2 lambda**: 0.01 (default), try 0.001 - 0.1
- **Dropout**: 0.2 (default), try 0.0 - 0.5
- **Batch size**: 32, try 16 - 128
- **Architecture**: Start with [input, 32, 16, 1], experiment with depth/width

### Training Tips:
1. **Start simple**: Use Adam with default settings
2. **Add regularization**: If overfitting, increase dropout/L2
3. **Adjust architecture**: If underfitting, add layers/neurons
4. **Use early stopping**: Prevents overfitting and saves time
5. **Monitor both losses**: Training should decrease, validation should stabilize

## Mathematical Details

### Adam Update Rule:
```
m_t = β₁ * m_{t-1} + (1 - β₁) * gradient
v_t = β₂ * v_{t-1} + (1 - β₂) * gradient²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
```

Where:
- β₁ = 0.9 (first moment decay)
- β₂ = 0.999 (second moment decay)
- α = learning rate
- ε = 1e-8 (numerical stability)

### L2 Regularization:
```
Loss_total = Loss_BCE + (λ / 2m) * Σ(w²)
Gradient_w = Gradient_BCE + (λ / m) * w
```

## Performance Comparison

Expected improvements with Adam vs SGD:
- **Convergence speed**: 2-5x faster
- **Final accuracy**: 0-5% better
- **Stability**: More robust to hyperparameter choices
