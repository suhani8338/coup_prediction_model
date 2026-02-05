"""
Systematic Neural Network Improvement Experiments
"""

import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.train_neural_network import train_and_evaluate, plot_results


def experiment_1_architecture(X_train, X_test, y_train, y_test):
    """Experiment with different architectures"""
    print("\n" + "="*80)
    print("EXPERIMENT 1: ARCHITECTURE SEARCH")
    print("="*80)
    
    architectures = {
        'Baseline (shallow)': [14, 32, 16, 1],
        'Deeper': [14, 64, 64, 32, 1],
        'Wider': [14, 128, 64, 32, 1],
        'Very Deep': [14, 64, 64, 64, 32, 1],
        'Wide + Deep': [14, 128, 128, 64, 32, 1],
        'Custom 1': [14, 96, 48, 24, 1],
        'Custom 2': [14, 80, 80, 40, 1],
    }
    
    results = {}
    
    for name, arch in architectures.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name} - {arch}")
        print(f"{'='*80}")
        
        model, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            layer_sizes=arch,
            learning_rate=0.001,
            optimizer='adam',
            l2_lambda=0.01,
            dropout_rate=0.2,
            epochs=500,
            batch_size=32,
            verbose=False,
            early_stopping_patience=30
        )
        
        results[name] = {
            'architecture': arch,
            'test_f1': metrics['test']['f1'],
            'test_acc': metrics['test']['accuracy'],
            'train_acc': metrics['train']['accuracy']
        }
    
    # Print summary
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Architecture':<25} {'Layers':<30} {'Test F1':<12} {'Test Acc':<12} {'Overfit Gap':<12}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    for name, result in sorted_results:
        arch_str = str(result['architecture'])
        overfit_gap = result['train_acc'] - result['test_acc']
        print(f"{name:<25} {arch_str:<30} {result['test_f1']:<12.4f} {result['test_acc']:<12.4f} {overfit_gap:<12.4f}")
    
    best = sorted_results[0]
    print(f"\n🏆 BEST ARCHITECTURE: {best[0]} with F1={best[1]['test_f1']:.4f}")
    
    return results


def experiment_2_learning_rate(X_train, X_test, y_train, y_test, best_arch=None):
    """Experiment with learning rates and schedulers"""
    print("\n" + "="*80)
    print("EXPERIMENT 2: LEARNING RATE OPTIMIZATION")
    print("="*80)
    
    if best_arch is None:
        best_arch = [14, 128, 64, 32, 1]  # Default to wider architecture
    
    lr_configs = {
        'Baseline (0.001)': {'lr': 0.001, 'decay': None},
        'Higher (0.005)': {'lr': 0.005, 'decay': None},
        'Higher (0.01)': {'lr': 0.01, 'decay': None},
        'LR with decay (0.01)': {'lr': 0.01, 'decay': 0.95},
        'LR with decay (0.005)': {'lr': 0.005, 'decay': 0.98},
        'Lower (0.0005)': {'lr': 0.0005, 'decay': None},
        'Aggressive decay': {'lr': 0.02, 'decay': 0.90},
    }
    
    results = {}
    
    for name, config in lr_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        model, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            layer_sizes=best_arch,
            learning_rate=config['lr'],
            optimizer='adam',
            l2_lambda=0.01,
            dropout_rate=0.2,
            epochs=500,
            batch_size=32,
            verbose=False,
            early_stopping_patience=30,
            lr_decay=config['decay']
        )
        
        results[name] = {
            'config': config,
            'test_f1': metrics['test']['f1'],
            'test_acc': metrics['test']['accuracy']
        }
    
    # Print summary
    print("\n" + "="*80)
    print("LEARNING RATE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Configuration':<30} {'LR':<12} {'Decay':<12} {'Test F1':<12} {'Test Acc':<12}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    for name, result in sorted_results:
        lr = result['config']['lr']
        decay = result['config']['decay'] if result['config']['decay'] else 'None'
        print(f"{name:<30} {lr:<12.4f} {str(decay):<12} {result['test_f1']:<12.4f} {result['test_acc']:<12.4f}")
    
    best = sorted_results[0]
    print(f"\n🏆 BEST LR CONFIG: {best[0]} with F1={best[1]['test_f1']:.4f}")
    
    return results


def experiment_3_regularization(X_train, X_test, y_train, y_test, best_arch=None, best_lr=0.001):
    """Experiment with regularization (L2 and Dropout)"""
    print("\n" + "="*80)
    print("EXPERIMENT 3: REGULARIZATION TUNING")
    print("="*80)
    
    if best_arch is None:
        best_arch = [14, 128, 64, 32, 1]
    
    reg_configs = {
        'No regularization': {'l2': 0.0, 'dropout': 0.0},
        'Light L2': {'l2': 0.001, 'dropout': 0.0},
        'Medium L2': {'l2': 0.01, 'dropout': 0.0},
        'Light dropout': {'l2': 0.0, 'dropout': 0.2},
        'Medium dropout': {'l2': 0.0, 'dropout': 0.3},
        'Heavy dropout': {'l2': 0.0, 'dropout': 0.5},
        'Balanced (light)': {'l2': 0.001, 'dropout': 0.2},
        'Balanced (medium)': {'l2': 0.01, 'dropout': 0.3},
        'Light both': {'l2': 0.001, 'dropout': 0.1},
    }
    
    results = {}
    
    for name, config in reg_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        model, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            layer_sizes=best_arch,
            learning_rate=best_lr,
            optimizer='adam',
            l2_lambda=config['l2'],
            dropout_rate=config['dropout'],
            epochs=500,
            batch_size=32,
            verbose=False,
            early_stopping_patience=30
        )
        
        results[name] = {
            'config': config,
            'test_f1': metrics['test']['f1'],
            'test_acc': metrics['test']['accuracy'],
            'train_acc': metrics['train']['accuracy']
        }
    
    # Print summary
    print("\n" + "="*80)
    print("REGULARIZATION COMPARISON RESULTS")
    print("="*80)
    print(f"{'Configuration':<25} {'L2':<12} {'Dropout':<12} {'Test F1':<12} {'Overfit Gap':<12}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    for name, result in sorted_results:
        l2 = result['config']['l2']
        dropout = result['config']['dropout']
        overfit_gap = result['train_acc'] - result['test_acc']
        print(f"{name:<25} {l2:<12.4f} {dropout:<12.4f} {result['test_f1']:<12.4f} {overfit_gap:<12.4f}")
    
    best = sorted_results[0]
    print(f"\n🏆 BEST REG CONFIG: {best[0]} with F1={best[1]['test_f1']:.4f}")
    
    return results


def experiment_4_batch_size(X_train, X_test, y_train, y_test, best_arch=None, best_lr=0.001, best_l2=0.01, best_dropout=0.2):
    """Experiment with batch sizes"""
    print("\n" + "="*80)
    print("EXPERIMENT 4: BATCH SIZE OPTIMIZATION")
    print("="*80)
    
    if best_arch is None:
        best_arch = [14, 128, 64, 32, 1]
    
    batch_sizes = [8, 16, 32, 64, 128]
    results = {}
    
    for bs in batch_sizes:
        print(f"\n{'='*80}")
        print(f"Testing batch size: {bs}")
        print(f"{'='*80}")
        
        model, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            layer_sizes=best_arch,
            learning_rate=best_lr,
            optimizer='adam',
            l2_lambda=best_l2,
            dropout_rate=best_dropout,
            epochs=500,
            batch_size=bs,
            verbose=False,
            early_stopping_patience=30
        )
        
        results[f'Batch {bs}'] = {
            'batch_size': bs,
            'test_f1': metrics['test']['f1'],
            'test_acc': metrics['test']['accuracy']
        }
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH SIZE COMPARISON RESULTS")
    print("="*80)
    print(f"{'Configuration':<20} {'Batch Size':<15} {'Test F1':<12} {'Test Acc':<12}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_f1'], reverse=True)
    for name, result in sorted_results:
        print(f"{name:<20} {result['batch_size']:<15} {result['test_f1']:<12.4f} {result['test_acc']:<12.4f}")
    
    best = sorted_results[0]
    print(f"\n🏆 BEST BATCH SIZE: {best[0]} with F1={best[1]['test_f1']:.4f}")
    
    return results


def run_all_experiments(quick_mode=False):
    """Run all experiments systematically"""
    print("="*80)
    print("NEURAL NETWORK IMPROVEMENT - SYSTEMATIC EXPERIMENTS")
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    X_train = np.load(os.path.join(project_root, 'data', 'X_train.npy'))
    X_test = np.load(os.path.join(project_root, 'data', 'X_test.npy'))
    y_train = np.load(os.path.join(project_root, 'data', 'y_train.npy'))
    y_test = np.load(os.path.join(project_root, 'data', 'y_test.npy'))
    
    print(f"✓ Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    # Experiment 1: Architecture
    arch_results = experiment_1_architecture(X_train, X_test, y_train, y_test)
    best_arch_name = max(arch_results.items(), key=lambda x: x[1]['test_f1'])[0]
    best_arch = arch_results[best_arch_name]['architecture']
    
    if quick_mode:
        print(f"\n⚡ Quick mode - skipping remaining experiments")
        print(f"Recommended architecture: {best_arch}")
        return
    
    # Experiment 2: Learning Rate (using best architecture)
    lr_results = experiment_2_learning_rate(X_train, X_test, y_train, y_test, best_arch)
    best_lr_name = max(lr_results.items(), key=lambda x: x[1]['test_f1'])[0]
    best_lr = lr_results[best_lr_name]['config']['lr']
    
    # Experiment 3: Regularization (using best arch + lr)
    reg_results = experiment_3_regularization(X_train, X_test, y_train, y_test, best_arch, best_lr)
    best_reg_name = max(reg_results.items(), key=lambda x: x[1]['test_f1'])[0]
    best_l2 = reg_results[best_reg_name]['config']['l2']
    best_dropout = reg_results[best_reg_name]['config']['dropout']
    
    # Experiment 4: Batch Size (using all best params)
    bs_results = experiment_4_batch_size(X_train, X_test, y_train, y_test, best_arch, best_lr, best_l2, best_dropout)
    best_bs_name = max(bs_results.items(), key=lambda x: x[1]['test_f1'])[0]
    best_bs = bs_results[best_bs_name]['batch_size']
    
    # Final Summary
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("="*80)
    print(f"Best Architecture:     {best_arch}")
    print(f"Best Learning Rate:    {best_lr}")
    print(f"Best L2 Lambda:        {best_l2}")
    print(f"Best Dropout Rate:     {best_dropout}")
    print(f"Best Batch Size:       {best_bs}")
    print("="*80)
    
    # Train final model with best params
    print("\n🚀 Training FINAL MODEL with optimized hyperparameters...")
    final_model, final_metrics = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        layer_sizes=best_arch,
        learning_rate=best_lr,
        optimizer='adam',
        l2_lambda=best_l2,
        dropout_rate=best_dropout,
        epochs=2000,
        batch_size=best_bs,
        verbose=True,
        early_stopping_patience=60
    )
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    
    return final_model, final_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Network Improvement Experiments')
    parser.add_argument('--quick', action='store_true', help='Run only architecture search (quick mode)')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4], help='Run specific experiment (1-4)')
    
    args = parser.parse_args()
    
    if args.exp:
        # Run specific experiment
        X_train = np.load(os.path.join(project_root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(project_root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(project_root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(project_root, 'data', 'y_test.npy'))
        
        if args.exp == 1:
            experiment_1_architecture(X_train, X_test, y_train, y_test)
        elif args.exp == 2:
            experiment_2_learning_rate(X_train, X_test, y_train, y_test)
        elif args.exp == 3:
            experiment_3_regularization(X_train, X_test, y_train, y_test)
        elif args.exp == 4:
            experiment_4_batch_size(X_train, X_test, y_train, y_test)
    else:
        # Run all experiments
        run_all_experiments(quick_mode=args.quick)
