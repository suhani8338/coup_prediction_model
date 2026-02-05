# Neural Network Model Improvements Summary

## What Was Improved

### 0. **Data Preprocessing & Feature Selection** 🔧
Critical data quality improvements that directly impact model performance.

**Data Leakage Removed:**
- Identified and removed 5 outcome-dependent variables that were leaking target information
- `unrealized`, `conspiracy`, `attempt`: Event outcome classifications (describe what happened, not predictors)
- `coup_id`, `event_type`: Non-informative identifiers
- Result: CV accuracy dropped from suspicious 100% to realistic 72.68%

**Feature Set:**
- **Final features**: 14 predictive features (no outcome variables)
- **Dataset**: 981 samples → Training: 784×14, Test: 197×14
- **Features kept**: All coup characteristics despite low correlations
  - Coup actor types: military, dissident, rebel, palace, foreign
  - Coup characteristics: auto, popular, counter, resign, other
  - Temporal/contextual: year_norm, month_sin, month_cos, country_freq

**Why keep low-correlation features:**
- Neural networks learn non-linear interactions (correlation only measures linear relationships)
- Domain knowledge: "who" and "how" of coups matter for prediction
- Random Forest analysis showed ALL features contribute (importance >0.5%)
- Small dataset (981 samples) benefits from retaining informative features

**Documentation**: See [FEATURE_SELECTION_ANALYSIS.md](FEATURE_SELECTION_ANALYSIS.md) for detailed analysis

### 1. **Adam Optimizer (Manually Implemented)** ✨
The crown jewel of these improvements. Adam is significantly better than vanilla SGD.

**Implementation Details:**
```python
# Adam maintains two moving averages per parameter:
m_t = β₁ * m_{t-1} + (1 - β₁) * gradient        # First moment (momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * gradient²       # Second moment (adaptive LR)

# Bias correction (important in early training):
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# Weight update:
w_t = w_{t-1} - α * m̂_t / (√v̂_t + ε)
```

**Why it's better:**
- Combines momentum (speeds up training in relevant directions)
- Adapts learning rate per parameter (handles sparse gradients)
- Requires less hyperparameter tuning
- Typical speedup: 2-5x faster convergence

### 2. **L2 Regularization (Ridge)**
Prevents overfitting by penalizing large weights.

```python
Loss_total = BCE_Loss + (λ / 2m) * Σ(weights²)
```

**Effect:** Smoother decision boundaries, better generalization

### 3. **Dropout Regularization**
Randomly drops neurons during training (set to 0 with probability p).

**Benefits:**
- Forces network to learn redundant representations
- Prevents co-adaptation of neurons
- Acts like ensemble learning

### 4. **Early Stopping**
Monitors validation loss and stops when no improvement for N epochs.

**Advantages:**
- Prevents overfitting
- Saves training time
- Automatically finds optimal number of epochs

### 5. **Learning Rate Decay**
Gradually reduces learning rate: `lr_t = lr_0 * (decay_rate)^t`

**Benefit:** Fine-tunes weights in later epochs

### 6. **Multiple Optimizers**
- **SGD**: Baseline
- **RMSprop**: Adaptive learning rate (middle ground)
- **Adam**: Best default choice

## How to Use

### Basic Usage (Adam with all improvements):
```python
from models.neural_network import NeuralNetwork

model = NeuralNetwork(
    layer_sizes=[15, 32, 16, 1],
    learning_rate=0.001,        # Lower for Adam
    optimizer='adam',           # Key improvement!
    l2_lambda=0.01,            # Regularization
    dropout_rate=0.2           # 20% dropout
)

model.fit(
    X_train, y_train,
    epochs=2000,
    batch_size=32,
    X_val=X_val,
    y_val=y_val,
    early_stopping_patience=50,
    lr_decay=0.95              # Optional
)
```

### Using the Training Script:
```python
from models.train_neural_network import train_and_evaluate, plot_results

model, metrics = train_and_evaluate(
    X_train, X_test, y_train, y_test,
    optimizer='adam',
    learning_rate=0.001,
    l2_lambda=0.01,
    dropout_rate=0.2,
    epochs=1000,
    early_stopping_patience=50
)

plot_results(model, metrics)
```

### Compare Optimizers:
```bash
python models/compare_models.py
```

## Expected Performance Improvements

| Metric | Basic SGD | With Adam + Regularization |
|--------|-----------|---------------------------|
| Training Speed | Baseline | 2-5x faster |
| Convergence | ~1000 epochs | ~200-500 epochs |
| Overfitting | High risk | Reduced |
| Hyperparameter Sensitivity | High | Low |
| Final Accuracy | Variable | More stable |

**Realistic Performance Expectations:**
- **Baseline** (Random Forest with clean data): 72.68% ± 2.25%
- **Neural Network Goal**: Match or exceed 72-75% accuracy
- **Note**: After removing data leakage, 70-75% is realistic for this challenging task
- **Class Distribution**: ~45% success, ~55% failure (slightly imbalanced)

## Hyperparameter Recommendations

### For Adam:
```python
learning_rate = 0.001       # Default, usually works well
l2_lambda = 0.01           # Start here, increase if overfitting
dropout_rate = 0.2         # 20% is a good starting point
batch_size = 32            # Standard choice
early_stopping_patience = 50  # ~10-20% of total epochs
```

### For SGD (if needed):
```python
learning_rate = 0.01       # Needs higher LR than Adam
l2_lambda = 0.001          # Start lower
dropout_rate = 0.3         # Can use more dropout
```

## File Structure

```
models/
├── neural_network.py          # Core NN class with Adam
├── train_neural_network.py    # Training and evaluation
├── compare_models.py          # Compare optimizers/hyperparameters
└── README.md                  # Detailed model documentation

data/
├── Coup data 2.1.2.csv       # Original dataset
├── X_train.npy               # Clean training features (784×14)
├── X_test.npy                # Clean test features (197×14)
├── y_train.npy               # Training labels (784,)
└── y_test.npy                # Test labels (197,)

notebooks/
└── eda.ipynb                 # EDA with feature selection analysis

FEATURE_SELECTION_ANALYSIS.md  # Data leakage & feature selection details
IMPROVEMENTS.md               # This file - model improvements summary
```

## Quick Reference

**To train a model:**
```bash
cd models
python train_neural_network.py
```

**To compare optimizers:**
```bash
python compare_models.py
```

**In a notebook:**
```python
%run ../models/train_neural_network.py
```

## Key Takeaways

1. **Always start with Adam** - it's the best default optimizer
2. **Use early stopping** - saves time and prevents overfitting
3. **Lower learning rate for Adam** - 0.001 vs 0.01 for SGD
4. **Add regularization** - L2 + dropout work well together
5. **Monitor both losses** - training loss should decrease, validation should stabilize

## Mathematical Intuition

### Why Adam Works:
1. **Momentum** (first moment): Accelerates in consistent directions
2. **Adaptive LR** (second moment): Bigger updates for infrequent features
3. **Bias correction**: Ensures proper initialization

Think of it as a ball rolling down a hill:
- Momentum: Ball gains speed going downhill
- Adaptive LR: Automatically adjusts to terrain
- Bias correction: Proper initial push

### Regularization Intuition:
- **L2**: "Keep weights small and similar"
- **Dropout**: "Learn multiple ways to solve the problem"
- **Early stopping**: "Stop before memorizing"

Together they prevent the model from being too confident about training data patterns that don't generalize.

---

**Next Steps:**
1. ✅ **Data cleaned**: Removed outcome-dependent variables (see FEATURE_SELECTION_ANALYSIS.md)
2. Run `train_neural_network.py` to train on clean 14-feature dataset
3. Try `compare_models.py` to compare optimizers
4. Aim to match/exceed 72.68% baseline (Random Forest)
5. Experiment with hyperparameters and architectures
6. Evaluate with precision/recall (not just accuracy) due to class imbalance
7. Consider ensemble methods combining multiple models
