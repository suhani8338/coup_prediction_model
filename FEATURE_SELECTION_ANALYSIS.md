# Feature Selection Analysis & Recommendations

## Executive Summary

After analyzing correlation values and understanding the codebook context, **DO NOT drop features with low correlation (-0.09 to 0.09)**. Instead, the critical issue is **data leakage** from outcome-dependent variables that must be removed.

## 🚨 Critical Finding: Data Leakage

### Variables REMOVED (Outcome-Dependent):
1. **`unrealized`** - Perfect inverse of target (-1.0 correlation)
2. **`conspiracy`** - Event outcome classification (coup plot discovered)
3. **`attempt`** - Event outcome classification (coup attempted but failed)
4. **`coup_id`** - Non-informative identifier
5. **`event_type`** - String representation of event classification

These variables describe **what happened** to the coup (the outcome), not characteristics that could predict success.

## ✅ Final Feature Set (14 Features)

### Coup Actor Types (Who Initiated - 5 features):
- `military`: Military-led coup
- `dissident`: Led by political dissidents
- `rebel`: Led by rebel forces
- `palace`: Internal palace coup
- `foreign`: Foreign-backed coup

### Coup Characteristics (How Executed - 5 features):
- `auto`: Auto-coup (leader stages coup against own government)
- `popular`: Popular uprising component
- `counter`: Counter-coup (response to previous coup)
- `resign`: Leader resignation involved
- `other`: Other characteristics

### Temporal & Contextual (4 features):
- `year_norm`: Normalized year (time trend)
- `month_sin`: Month (cyclical encoding - sine)
- `month_cos`: Month (cyclical encoding - cosine)
- `country_freq`: Country's historical coup frequency

## 📊 Correlation Analysis Results

### Features with Low Correlation BUT High Model Importance:

| Feature | Correlation | RF Importance | Keep? |
|---------|-------------|---------------|-------|
| `military` | 0.0097 | 0.0371 | ✅ YES |
| `month_sin` | -0.0354 | 0.0269 | ✅ YES |
| `month_cos` | 0.0109 | 0.0209 | ✅ YES |
| `foreign` | -0.0089 | 0.0069 | ✅ YES |
| `counter` | 0.0726 | 0.0097 | ✅ YES |
| `year_norm` | -0.0686 | 0.0711 | ✅ YES |
| `country_freq` | -0.0105 | 0.0385 | ✅ YES |

**Key Insight**: Features like `military` have near-zero correlation (0.0097) but contribute 3.7% to model importance. This demonstrates why correlation-based filtering would be harmful.

## 🎯 Why NOT to Drop Low-Correlation Features

### 1. **Neural Networks Learn Non-Linear Patterns**
- Individual correlations measure linear relationships only
- Features interact in complex ways (e.g., military + palace coup combinations)
- Neural networks excel at discovering these interactions

### 2. **Domain Knowledge is Critical**
- Coup type (military vs. dissident vs. rebel) is theoretically important
- The **who** and **how** of a coup matters for predicting success
- Low correlation ≠ low predictive value

### 3. **Small Dataset (981 samples)**
- Limited data means every informative feature helps
- Risk of overfitting increases when features are too sparse
- Better to use regularization than arbitrary feature removal

### 4. **Empirical Evidence**
- Random Forest importance shows ALL features contribute
- Cross-validation accuracy: 72.68% (realistic performance)
- Removing features with importance >1% would discard 11/14 features

## 📈 Model Performance

### Before Removing Data Leakage:
- CV Accuracy: **100%** (suspicious!)
- Cause: `unrealized`, `conspiracy`, `attempt` leak target information

### After Removing Data Leakage:
- CV Accuracy: **72.68% ± 2.25%**
- Realistic performance for a challenging prediction task
- Class distribution: ~45% success, ~55% failure

## 🔧 Implementation Changes

### Updated Data Pipeline:
```python
# BEFORE (with data leakage):
X = df.drop(columns=["realized"])
# Includes: unrealized, conspiracy, attempt ❌

# AFTER (clean):
outcome_vars = ['unrealized', 'conspiracy', 'attempt']
X = df.drop(columns=["realized"] + outcome_vars)
# Only true predictive features ✅
```

### New Dataset Dimensions:
- **Training set**: 784 samples × 14 features
- **Test set**: 197 samples × 14 features
- **Features**: All coup characteristics (no outcome variables)

## 🎓 Lessons Learned from Codebook

### Event Classification (Mutually Exclusive):
The dataset categorizes coup events into:
1. **Realized** (target): Successful regime change
2. **Unrealized**: Failed, no regime change
3. **Conspiracy**: Plot discovered before execution
4. **Attempt**: Attempted but failed

These are **outcome labels**, not independent predictors!

### Coup Success Rates by Type:
- **Popular uprisings**: 81% success rate (highest correlation: 0.254)
- **Auto-coups**: 74% success rate
- **Palace coups**: 72% success rate
- **Counter-coups**: 71% success rate
- **Military coups**: 47% success rate
- **Dissident-led**: 9% success rate (strong negative correlation: -0.478)

## ✅ Final Recommendations

### 1. Feature Selection Strategy:
- ✅ **KEEP** all 14 clean features (no correlation-based filtering)
- ❌ **REMOVE** outcome-dependent variables (unrealized, conspiracy, attempt)
- ✅ **USE** regularization (dropout, L1/L2) instead of manual feature selection

### 2. Model Training:
- Use the cleaned dataset (14 features, no leakage)
- Expect realistic accuracy (~70-75% range)
- Monitor for overfitting with proper validation

### 3. Feature Engineering Opportunities:
- Coup type combinations (e.g., military + palace)
- Country-specific historical patterns
- Time-based features (decade indicators)
- Regional indicators (if country data available)

### 4. Evaluation:
- Baseline accuracy: 72.68% (Random Forest with regularization)
- Neural network should aim to match or exceed this
- Use stratified K-fold cross-validation
- Report precision/recall (not just accuracy) due to class imbalance

## 📁 Updated Files

- **Data**: [data/X_train.npy](data/X_train.npy) - 784×14 (clean features)
- **Data**: [data/X_test.npy](data/X_test.npy) - 197×14 (clean features)
- **Notebook**: [notebooks/eda.ipynb](notebooks/eda.ipynb) - Updated with analysis

## 🔍 Next Steps

1. ✅ Retrain neural network with clean 14-feature dataset
2. ✅ Compare performance to 72.68% Random Forest baseline
3. ✅ Implement proper evaluation metrics (precision, recall, F1)
4. ✅ Consider ensemble methods combining multiple models
5. ✅ Document realistic expectations (70-75% accuracy range)
