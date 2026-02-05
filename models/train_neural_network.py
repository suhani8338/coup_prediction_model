"""
Training and evaluation script for Neural Network model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from neural_network import NeuralNetwork


def train_and_evaluate(X_train, X_test, y_train, y_test, 
                       layer_sizes=None, learning_rate=0.001, optimizer='adam',
                       l2_lambda=0.01, dropout_rate=0.2,
                       epochs=1000, batch_size=32, verbose=True,
                       early_stopping_patience=50, lr_decay=None):
    """
    Train and evaluate a neural network model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    X_test : numpy.ndarray
        Test features
    y_train : numpy.ndarray
        Training labels
    y_test : numpy.ndarray
        Test labels
    layer_sizes : list, optional
        Network architecture. If None, uses [input_size, 32, 16, 1]
    learning_rate : float
        Learning rate for gradient descent (default: 0.001 for Adam)
    optimizer : str
        Optimizer to use: 'sgd', 'adam', 'rmsprop' (default: 'adam')
    l2_lambda : float
        L2 regularization parameter (default: 0.01)
    dropout_rate : float
        Dropout rate for hidden layers (default: 0.2)
    epochs : int
        Number of training epochs
    batch_size : int
        Mini-batch size
    verbose : bool
        Whether to print training progress
    early_stopping_patience : int, optional
        Number of epochs to wait for improvement before stopping
    lr_decay : float, optional
        Learning rate decay factor per epoch
        
    Returns:
    --------
    tuple
        (trained_model, metrics_dict)
    """
    # Split training into train/validation
    val_split = 0.1
    val_size = int(len(X_train) * val_split)
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train_sub = X_train[val_size:]
    y_train_sub = y_train[val_size:]
    
    # Ensure y arrays are 2D for neural network (reshape if needed)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_val.ndim == 1:
        y_val = y_val.reshape(-1, 1)
    if y_train_sub.ndim == 1:
        y_train_sub = y_train_sub.reshape(-1, 1)
    
    # Determine architecture
    if layer_sizes is None:
        input_size = X_train.shape[1]
        layer_sizes = [input_size, 32, 16, 1]
    
    print(f"Network architecture: {layer_sizes}")
    print(f"Optimizer: {optimizer.upper()}")
    print(f"Learning rate: {learning_rate}")
    print(f"L2 regularization: {l2_lambda}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("-" * 60)
    
    # Create and train model
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        optimizer=optimizer,
        l2_lambda=l2_lambda,
        dropout_rate=dropout_rate
    )
    
    nn.fit(X_train_sub, y_train_sub, 
           epochs=epochs, batch_size=batch_size, verbose=verbose,
           X_val=X_val, y_val=y_val,
           early_stopping_patience=early_stopping_patience,
           lr_decay=lr_decay)
    
    # Make predictions
    y_train_pred = nn.predict(X_train)
    y_test_pred = nn.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("=== Training Set Performance ===")
    print(f"Accuracy:  {metrics['train']['accuracy']:.4f}")
    print(f"Precision: {metrics['train']['precision']:.4f}")
    print(f"Recall:    {metrics['train']['recall']:.4f}")
    print(f"F1-Score:  {metrics['train']['f1']:.4f}")
    
    print("\n=== Test Set Performance ===")
    print(f"Accuracy:  {metrics['test']['accuracy']:.4f}")
    print(f"Precision: {metrics['test']['precision']:.4f}")
    print(f"Recall:    {metrics['test']['recall']:.4f}")
    print(f"F1-Score:  {metrics['test']['f1']:.4f}")
    
    print("\n=== Confusion Matrix (Test Set) ===")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    print("\n=== Classification Report (Test Set) ===")
    print(classification_report(y_test, y_test_pred, target_names=['Failed', 'Realized']))
    print("=" * 60)
    
    # Store predictions and confusion matrix for plotting
    metrics['test_predictions'] = y_test_pred
    metrics['confusion_matrix'] = cm
    
    return nn, metrics


def plot_results(model, metrics, save_path=None):
    """
    Plot training loss and confusion matrix.
    
    Parameters:
    -----------
    model : NeuralNetwork
        Trained neural network model
    metrics : dict
        Dictionary containing metrics and predictions
    save_path : str, optional
        Path to save the plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    axes[0].plot(model.loss_history, label='Training Loss')
    if hasattr(model, 'val_loss_history') and model.val_loss_history:
        axes[0].plot(model.val_loss_history, label='Validation Loss', linestyle='--')
        axes[0].legend()
    axes[0].set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Cross-Entropy Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot confusion matrix
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Failed', 'Realized'],
                yticklabels=['Failed', 'Realized'])
    axes[1].set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    """
    Example usage when running this script directly.
    Expects preprocessed data files in the data directory.
    """
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    print("Loading preprocessed data...")
    
    # Load preprocessed data
    try:
        X_train = np.load(os.path.join(project_root, 'data', 'X_train.npy'))
        X_test = np.load(os.path.join(project_root, 'data', 'X_test.npy'))
        y_train = np.load(os.path.join(project_root, 'data', 'y_train.npy'))
        y_test = np.load(os.path.join(project_root, 'data', 'y_test.npy'))
        
        print(f"Data loaded successfully!")
        print(f"Training set: {X_train.shape}, {y_train.shape}")
        print(f"Test set: {X_test.shape}, {y_test.shape}")
        
        # Train and evaluate
        model, metrics = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            layer_sizes=None,  # Uses default [input_size, 32, 16, 1]]
            learning_rate=0.001,
            optimizer='adam',
            l2_lambda=0.01,
            dropout_rate=0.2,
            epochs=1500,
            batch_size=32,
            verbose=True,
            early_stopping_patience=50
        )
        
        # Plot results
        plot_results(model, metrics)
        
    except FileNotFoundError:
        print("Error: Preprocessed data files not found!")
        print("Please run the data preparation in the notebook first.")
        print("Expected files:")
        print("  - data/X_train.npy")
        print("  - data/X_test.npy")
        print("  - data/y_train.npy")
        print("  - data/y_test.npy")
