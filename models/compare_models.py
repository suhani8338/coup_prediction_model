"""
Model comparison script - comparing Neural Network with traditional ML models
"""

import numpy as np
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.train_neural_network import train_and_evaluate, plot_results


def compare_ml_models(X_train, X_test, y_train, y_test):
    """Compare Neural Network with traditional ML models"""
    print("=" * 80)
    print("COMPARING NEURAL NETWORK WITH TRADITIONAL ML MODELS")
    print("=" * 80)
    
    results = {}
    
    # 1. Logistic Regression
    print(f"\n{'='*80}")
    print("Training Logistic Regression")
    print(f"{'='*80}")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)
    
    results['Logistic Regression'] = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    # 2. Decision Tree
    print(f"\n{'='*80}")
    print("Training Decision Tree")
    print(f"{'='*80}")
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    
    y_train_pred = dt_model.predict(X_train)
    y_test_pred = dt_model.predict(X_test)
    
    results['Decision Tree'] = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    # 3. Random Forest
    print(f"\n{'='*80}")
    print("Training Random Forest")
    print(f"{'='*80}")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_train_pred = rf_model.predict(X_train)
    y_test_pred = rf_model.predict(X_test)
    
    results['Random Forest'] = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    # 4. SVM
    print(f"\n{'='*80}")
    print("Training SVM (RBF kernel)")
    print(f"{'='*80}")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train, y_train)
    
    y_train_pred = svm_model.predict(X_train)
    y_test_pred = svm_model.predict(X_test)
    
    results['SVM'] = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0)
    }
    
    # 5. Neural Network
    print(f"\n{'='*80}")
    print("Training Neural Network")
    print(f"{'='*80}")
    
    model, metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        optimizer='adam',
        learning_rate=0.001,
        l2_lambda=0.01,
        dropout_rate=0.2,
        epochs=500,
        batch_size=32,
        verbose=False,
        early_stopping_patience=30
    )
    
    results['Neural Network'] = {
        'train_accuracy': metrics['train']['accuracy'],
        'test_accuracy': metrics['test']['accuracy'],
        'test_precision': metrics['test']['precision'],
        'test_recall': metrics['test']['recall'],
        'test_f1': metrics['test']['f1']
    }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - ALL MODELS")
    print("=" * 80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<20} "
              f"{result['train_accuracy']:<12.4f} "
              f"{result['test_accuracy']:<12.4f} "
              f"{result['test_precision']:<12.4f} "
              f"{result['test_recall']:<12.4f} "
              f"{result['test_f1']:<12.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['test_f1'])
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model[0]} (F1 Score: {best_model[1]['test_f1']:.4f})")
    print("=" * 80)
    
    return results


def compare_optimizers(X_train, X_test, y_train, y_test):
    """Compare different optimizers on the same data"""
    print("=" * 80)
    print("COMPARING OPTIMIZERS")
    print("=" * 80)
    
    optimizers = {
        'sgd': {'learning_rate': 0.01},
        'adam': {'learning_rate': 0.001},
        'rmsprop': {'learning_rate': 0.001}
    }
    
    results = {}
    
    for opt_name, params in optimizers.items():
        print(f"\n{'='*80}")
        print(f"Training with {opt_name.upper()}")
        print(f"{'='*80}")
        
        model, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            optimizer=opt_name,
            learning_rate=params['learning_rate'],
            l2_lambda=0.01,
            dropout_rate=0.2,
            epochs=500,
            batch_size=32,
            verbose=False,
            early_stopping_patience=30
        )
        
        results[opt_name] = {
            'model': model,
            'metrics': metrics
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Optimizer':<15} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Epochs':<10}")
    print("-" * 80)
    
    for opt_name, result in results.items():
        metrics = result['metrics']
        epochs_trained = len(result['model'].loss_history)
        print(f"{opt_name.upper():<15} "
              f"{metrics['train']['accuracy']:<12.4f} "
              f"{metrics['test']['accuracy']:<12.4f} "
              f"{metrics['test']['f1']:<12.4f} "
              f"{epochs_trained:<10}")
    
    return results


def grid_search_hyperparameters(X_train, X_test, y_train, y_test):
    """Simple grid search for hyperparameters"""
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH (Adam Optimizer)")
    print("=" * 80)
    
    # Define grid
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'l2_lambda': [0.001, 0.01, 0.1],
        'dropout_rate': [0.0, 0.2, 0.5]
    }
    
    best_score = 0
    best_params = None
    all_results = []
    
    # Grid search
    for lr in param_grid['learning_rate']:
        for l2 in param_grid['l2_lambda']:
            for dropout in param_grid['dropout_rate']:
                print(f"\nTesting: LR={lr}, L2={l2}, Dropout={dropout}")
                
                model, metrics = train_and_evaluate(
                    X_train, X_test, y_train, y_test,
                    learning_rate=lr,
                    optimizer='adam',
                    l2_lambda=l2,
                    dropout_rate=dropout,
                    epochs=300,
                    batch_size=32,
                    verbose=False,
                    early_stopping_patience=20
                )
                
                test_f1 = metrics['test']['f1']
                all_results.append({
                    'params': {'lr': lr, 'l2': l2, 'dropout': dropout},
                    'test_f1': test_f1,
                    'test_acc': metrics['test']['accuracy']
                })
                
                print(f"  → Test F1: {test_f1:.4f}, Test Acc: {metrics['test']['accuracy']:.4f}")
                
                if test_f1 > best_score:
                    best_score = test_f1
                    best_params = {'lr': lr, 'l2': l2, 'dropout': dropout}
    
    # Print results
    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"Learning Rate: {best_params['lr']}")
    print(f"L2 Lambda: {best_params['l2']}")
    print(f"Dropout Rate: {best_params['dropout']}")
    print(f"Best Test F1 Score: {best_score:.4f}")
    
    # Print top 5
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)
    sorted_results = sorted(all_results, key=lambda x: x['test_f1'], reverse=True)[:5]
    
    for i, result in enumerate(sorted_results, 1):
        p = result['params']
        print(f"{i}. LR={p['lr']:.4f}, L2={p['l2']:.4f}, Dropout={p['dropout']:.2f} "
              f"→ F1={result['test_f1']:.4f}, Acc={result['test_acc']:.4f}")
    
    return best_params, all_results


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    try:
        X_train = np.load(os.path.join(project_root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(project_root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(project_root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(project_root, 'data', 'y_test.npy'))
        
        # Ensure y arrays are 1D (sklearn expects this)
        y_train = y_train.ravel() if y_train.ndim > 1 else y_train
        y_test = y_test.ravel() if y_test.ndim > 1 else y_test
        
        print(f"Data loaded: Train {X_train.shape}, Test {X_test.shape}\n")
        
        # Main comparison: Neural Network vs Traditional ML Models
        model_results = compare_ml_models(X_train, X_test, y_train, y_test)
        
        print("\n\n")
        
        # Optional: Compare different optimizers for Neural Network
        # Uncomment to run optimizer comparison
        # print("\n" + "="*80)
        # print("OPTIONAL: NEURAL NETWORK OPTIMIZER COMPARISON")
        # print("="*80)
        # optimizer_results = compare_optimizers(X_train, X_test, y_train, y_test)
        
        # Optional: Grid search (warning: takes longer)
        # Uncomment to run full grid search for Neural Network hyperparameters
        # best_params, all_results = grid_search_hyperparameters(X_train, X_test, y_train, y_test)
        
        print("\n✓ Model comparison complete!")
        
    except FileNotFoundError:
        print("Error: Data files not found. Run the preprocessing notebook first.")
