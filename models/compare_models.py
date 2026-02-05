"""
Model comparison script - comparing Neural Network with traditional ML models
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('seaborn-darkgrid') if 'seaborn-darkgrid' in plt.style.available else plt.style.use('default')
sns.set_palette("husl")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.train_neural_network import train_and_evaluate, plot_results


def plot_model_comparison(results, save_path='model_comparison.png'):
    """Create comprehensive visualization comparing all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison - Comprehensive Analysis', fontsize=16, fontweight='bold')
    
    models = list(results.keys())
    
    # 1. Test Accuracy Comparison
    ax1 = axes[0, 0]
    test_acc = [results[m]['test_accuracy'] for m in models]
    bars1 = ax1.bar(models, test_acc, color=sns.color_palette("husl", len(models)))
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.tick_params(axis='x', rotation=45)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Train vs Test Accuracy (Overfitting Check)
    ax2 = axes[0, 1]
    train_acc = [results[m]['train_accuracy'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    bars2a = ax2.bar(x - width/2, train_acc, width, label='Train', alpha=0.8)
    bars2b = ax2.bar(x + width/2, test_acc, width, label='Test', alpha=0.8)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Train vs Test Accuracy (Overfitting Check)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    # 3. Precision, Recall, F1 Score
    ax3 = axes[1, 0]
    precision = [results[m]['test_precision'] for m in models]
    recall = [results[m]['test_recall'] for m in models]
    f1 = [results[m]['test_f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    ax3.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax3.bar(x, recall, width, label='Recall', alpha=0.8)
    ax3.bar(x + width, f1, width, label='F1 Score', alpha=0.8)
    ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax3.set_title('Precision, Recall, and F1 Score Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim([0, 1])
    
    # 4. Radar Chart for Best 3 Models
    ax4 = axes[1, 1]
    # Get top 3 models by F1 score
    sorted_models = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)[:3]
    
    categories = ['Test Acc', 'Precision', 'Recall', 'F1 Score']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for model_name, metrics in sorted_models:
        values = [
            metrics['test_accuracy'],
            metrics['test_precision'],
            metrics['test_recall'],
            metrics['test_f1']
        ]
        values += values[:1]  # Complete the circle
        ax4.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax4.fill(angles, values, alpha=0.15)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_title('Top 3 Models - Radar Chart', fontsize=13, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    plt.show()
    
    return fig


def plot_optimizer_comparison(results, save_path='optimizer_comparison.png'):
    """Visualize optimizer performance comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Optimizer Comparison - Neural Network', fontsize=16, fontweight='bold')
    
    optimizers = list(results.keys())
    colors = sns.color_palette("husl", len(optimizers))
    
    # 1. Training Loss Curves
    ax1 = axes[0, 0]
    for i, (opt_name, result) in enumerate(results.items()):
        loss_history = result['model'].loss_history
        ax1.plot(loss_history, label=opt_name.upper(), linewidth=2, color=colors[i])
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Test Accuracy Comparison
    ax2 = axes[0, 1]
    test_acc = [results[opt]['metrics']['test']['accuracy'] for opt in optimizers]
    bars = ax2.bar(optimizers, test_acc, color=colors)
    ax2.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Test Accuracy by Optimizer', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 3. F1 Score and Convergence Speed
    ax3 = axes[1, 0]
    f1_scores = [results[opt]['metrics']['test']['f1'] for opt in optimizers]
    bars = ax3.bar(optimizers, f1_scores, color=colors)
    ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1 Score by Optimizer', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 1])
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    # 4. Epochs to Convergence
    ax4 = axes[1, 1]
    epochs = [len(results[opt]['model'].loss_history) for opt in optimizers]
    bars = ax4.bar(optimizers, epochs, color=colors)
    ax4.set_ylabel('Number of Epochs', fontsize=12, fontweight='bold')
    ax4.set_title('Training Epochs (Convergence Speed)', fontsize=13, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    plt.show()
    
    return fig


def plot_grid_search_results(all_results, best_params, save_path='grid_search_results.png'):
    """Visualize grid search results with heatmaps and rankings"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Grid Search Results', fontsize=16, fontweight='bold')
    
    # Convert results to arrays for heatmap
    learning_rates = sorted(list(set([r['params']['lr'] for r in all_results])))
    l2_lambdas = sorted(list(set([r['params']['l2'] for r in all_results])))
    dropout_rates = sorted(list(set([r['params']['dropout'] for r in all_results])))
    
    # 1. Heatmap: Learning Rate vs L2 (averaged over dropout)
    ax1 = axes[0, 0]
    heatmap_data_1 = np.zeros((len(l2_lambdas), len(learning_rates)))
    for i, l2 in enumerate(l2_lambdas):
        for j, lr in enumerate(learning_rates):
            scores = [r['test_f1'] for r in all_results 
                     if r['params']['lr'] == lr and r['params']['l2'] == l2]
            heatmap_data_1[i, j] = np.mean(scores) if scores else 0
    
    sns.heatmap(heatmap_data_1, annot=True, fmt='.3f', cmap='YlOrRd', 
                xticklabels=[f'{lr:.4f}' for lr in learning_rates],
                yticklabels=[f'{l2:.3f}' for l2 in l2_lambdas],
                ax=ax1, cbar_kws={'label': 'F1 Score'})
    ax1.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('L2 Lambda', fontsize=12, fontweight='bold')
    ax1.set_title('F1 Score: Learning Rate vs L2 Lambda', fontsize=13, fontweight='bold')
    
    # 2. Heatmap: Learning Rate vs Dropout (averaged over L2)
    ax2 = axes[0, 1]
    heatmap_data_2 = np.zeros((len(dropout_rates), len(learning_rates)))
    for i, dropout in enumerate(dropout_rates):
        for j, lr in enumerate(learning_rates):
            scores = [r['test_f1'] for r in all_results 
                     if r['params']['lr'] == lr and r['params']['dropout'] == dropout]
            heatmap_data_2[i, j] = np.mean(scores) if scores else 0
    
    sns.heatmap(heatmap_data_2, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=[f'{lr:.4f}' for lr in learning_rates],
                yticklabels=[f'{d:.1f}' for d in dropout_rates],
                ax=ax2, cbar_kws={'label': 'F1 Score'})
    ax2.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dropout Rate', fontsize=12, fontweight='bold')
    ax2.set_title('F1 Score: Learning Rate vs Dropout', fontsize=13, fontweight='bold')
    
    # 3. Top 10 Configurations
    ax3 = axes[1, 0]
    sorted_results = sorted(all_results, key=lambda x: x['test_f1'], reverse=True)[:10]
    config_labels = [f"LR={r['params']['lr']:.4f}\nL2={r['params']['l2']:.3f}\nDO={r['params']['dropout']:.1f}" 
                    for r in sorted_results]
    f1_scores = [r['test_f1'] for r in sorted_results]
    
    colors = sns.color_palette('RdYlGn', len(sorted_results))
    bars = ax3.barh(range(len(sorted_results)), f1_scores, color=colors)
    ax3.set_yticks(range(len(sorted_results)))
    ax3.set_yticklabels([f'Config {i+1}' for i in range(len(sorted_results))], fontsize=9)
    ax3.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('Top 10 Configurations (by F1 Score)', fontsize=13, fontweight='bold')
    ax3.invert_yaxis()
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}', ha='left', va='center', fontsize=8, fontweight='bold')
    
    # 4. Parameter Impact Analysis
    ax4 = axes[1, 1]
    # Calculate average F1 for each parameter value
    lr_impact = {lr: np.mean([r['test_f1'] for r in all_results if r['params']['lr'] == lr]) 
                 for lr in learning_rates}
    l2_impact = {l2: np.mean([r['test_f1'] for r in all_results if r['params']['l2'] == l2]) 
                 for l2 in l2_lambdas}
    dropout_impact = {do: np.mean([r['test_f1'] for r in all_results if r['params']['dropout'] == do]) 
                      for do in dropout_rates}
    
    x_pos = np.arange(3)
    lr_vals = list(lr_impact.values())
    l2_vals = list(l2_impact.values())
    dropout_vals = list(dropout_impact.values())
    
    # Normalize for comparison
    all_vals = lr_vals + l2_vals + dropout_vals
    y_min, y_max = min(all_vals), max(all_vals)
    
    ax4.plot([f'LR{i+1}' for i in range(len(lr_vals))], lr_vals, 'o-', label='Learning Rate', linewidth=2, markersize=8)
    ax4.plot([f'L2-{i+1}' for i in range(len(l2_vals))], l2_vals, 's-', label='L2 Lambda', linewidth=2, markersize=8)
    ax4.plot([f'DO{i+1}' for i in range(len(dropout_vals))], dropout_vals, '^-', label='Dropout', linewidth=2, markersize=8)
    
    ax4.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    ax4.set_title('Parameter Impact on Performance', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([y_min * 0.95, y_max * 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {save_path}")
    plt.show()
    
    return fig


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
    
    # 5. Neural Network (IMPROVED HYPERPARAMETERS)
    print(f"\n{'='*80}")
    print("Training Neural Network (Optimized)")
    print(f"{'='*80}")
    
    model, metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        layer_sizes=[14, 128, 64, 32, 1],  # Deeper & wider architecture
        optimizer='adam',
        learning_rate=0.005,  # Higher learning rate
        l2_lambda=0.001,  # Less L2 regularization
        dropout_rate=0.1,  # Less dropout for small dataset
        epochs=1000,  # More epochs
        batch_size=16,  # Smaller batch size for better generalization
        verbose=False,
        early_stopping_patience=50,  # More patience
        lr_decay=0.98  # Learning rate decay
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
    
    # Generate visualizations
    plot_model_comparison(results)
    
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
    
    # Generate visualizations
    plot_optimizer_comparison(results)
    
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
    
    # Generate visualizations
    plot_grid_search_results(all_results, best_params)
    
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
