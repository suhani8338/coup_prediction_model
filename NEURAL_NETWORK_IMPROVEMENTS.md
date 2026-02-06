# Neural Network Improvement Guide

## Goal
Beat baseline models (Logistic Regression, Decision Tree, Random Forest, SVM) with optimized neural network.

## Current Status
- **Current Architecture**: [14, 32, 16, 1] - relatively shallow
- **Training Samples**: 784 (small dataset - risk of overfitting)
- **Features**: 14

## Quick Wins (Try These First)

### 1. **Improve Architecture**
Current default is too shallow. Try:
```python
layer_sizes=[14, 128, 64, 32, 1]  # Wider + Deeper
# or
layer_sizes=[14, 96, 96, 48, 1]   # Balanced deep network
```

### 2. **Reduce Regularization**
For small datasets, less regularization often helps:
```python
l2_lambda=0.001  # Instead of 0.01
dropout_rate=0.1  # Instead of 0.2
```

### 3. **Tune Learning Rate with Decay**
```python
learning_rate=0.005
lr_decay=0.98  # Gradual decay improves convergence
```

### 4. **Smaller Batch Size**
With only 784 samples, use smaller batches:
```python
batch_size=16  # Instead of 32
```

### 5. **More Training Patience**
```python
epochs=1000
early_stopping_patience=75  # Instead of 30
```

---

## Systematic Approach

### Step 1: Run Quick Architecture Search
```bash
python models/improve_nn.py --quick
```

### Step 2: Run Full Optimization
```bash
python models/improve_nn.py
```

### Step 3: Compare with Baselines
Update `compare_models.py` with best params and run:
```bash
python models/compare_models.py
```

---

## Individual Experiments

Run specific experiments:
```bash
# Experiment 1: Architecture only
python models/improve_nn.py --exp 1

# Experiment 2: Learning rate only
python models/improve_nn.py --exp 2

# Experiment 3: Regularization only
python models/improve_nn.py --exp 3

# Experiment 4: Batch size only
python models/improve_nn.py --exp 4
```

---

## Files

- **improve_nn.py** - Systematic experiments script (NEW)
- **compare_models.py** - Compare NN with baseline models
- **train_neural_network.py** - Training utilities
- **neural_network.py** - NN implementation

---
