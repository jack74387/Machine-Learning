"""
Homework-2: Ensemble Learning (Chapter 7)
Multiclass Classification using MNIST and Fashion MNIST datasets
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# ============================================================================
# Part 1: The MNIST Dataset
# ============================================================================

def load_and_split_mnist():
    """
    Load MNIST dataset and split into training (50,000), validation (10,000), 
    and test (10,000) sets.
    """
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)
    
    # Split: 60,000 for train+val, 10,000 for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    # Split train_val: 50,000 for train, 10,000 for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=10000, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_individual_classifiers_mnist(X_train, y_train, X_val, y_val):
    """
    Train various classifiers: Random Forest, Extra Trees, and SVM.
    """
    print("\n" + "="*70)
    print("Training Individual Classifiers on MNIST")
    print("="*70)
    
    classifiers = {}
    
    # 1. Random Forest
    print("\n[1/3] Training Random Forest...")
    start = time.time()
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf_clf.predict(X_val))
    print(f"Random Forest - Validation Accuracy: {rf_val_acc:.4f} (Time: {time.time()-start:.2f}s)")
    classifiers['Random Forest'] = (rf_clf, rf_val_acc)
    
    # 2. Extra Trees
    print("\n[2/3] Training Extra Trees...")
    start = time.time()
    et_clf = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et_clf.fit(X_train, y_train)
    et_val_acc = accuracy_score(y_val, et_clf.predict(X_val))
    print(f"Extra Trees - Validation Accuracy: {et_val_acc:.4f} (Time: {time.time()-start:.2f}s)")
    classifiers['Extra Trees'] = (et_clf, et_val_acc)
    
    # 3. SVM (using linear kernel for speed)
    print("\n[3/3] Training SVM...")
    start = time.time()
    svm_clf = SVC(kernel='rbf', gamma='scale', random_state=42, probability=True)
    # Use subset for SVM due to computational cost
    svm_clf.fit(X_train[:10000], y_train[:10000])
    svm_val_acc = accuracy_score(y_val, svm_clf.predict(X_val))
    print(f"SVM - Validation Accuracy: {svm_val_acc:.4f} (Time: {time.time()-start:.2f}s)")
    classifiers['SVM'] = (svm_clf, svm_val_acc)
    
    return classifiers


def create_soft_voting_ensemble(classifiers):
    """
    Create an ensemble using soft voting.
    """
    print("\n" + "="*70)
    print("Creating Soft Voting Ensemble")
    print("="*70)
    
    estimators = [(name, clf) for name, (clf, _) in classifiers.items()]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    
    return voting_clf


def evaluate_ensemble_mnist(voting_clf, classifiers, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Train and evaluate the ensemble on validation and test sets.
    """
    print("\nTraining Soft Voting Ensemble...")
    start = time.time()
    voting_clf.fit(X_train, y_train)
    print(f"Training completed in {time.time()-start:.2f}s")
    
    # Validation accuracy
    val_pred = voting_clf.predict(X_val)
    ensemble_val_acc = accuracy_score(y_val, val_pred)
    print(f"\nEnsemble - Validation Accuracy: {ensemble_val_acc:.4f}")
    
    # Test accuracy
    test_pred = voting_clf.predict(X_test)
    ensemble_test_acc = accuracy_score(y_test, test_pred)
    print(f"Ensemble - Test Accuracy: {ensemble_test_acc:.4f}")
    
    # Compare with individual classifiers
    print("\n" + "="*70)
    print("Performance Comparison on Test Set")
    print("="*70)
    
    for name, (clf, val_acc) in classifiers.items():
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"{name:20s} - Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    
    print(f"{'Soft Voting Ensemble':20s} - Val: {ensemble_val_acc:.4f}, Test: {ensemble_test_acc:.4f}")
    
    # Calculate improvement
    best_individual_test = max([accuracy_score(y_test, clf.predict(X_test)) 
                                for _, (clf, _) in classifiers.items()])
    improvement = ensemble_test_acc - best_individual_test
    print(f"\nImprovement over best individual: {improvement:.4f} ({improvement*100:.2f}%)")
    
    return ensemble_val_acc, ensemble_test_acc


# ============================================================================
# Part 2: The Fashion MNIST Dataset
# ============================================================================

def load_and_split_fashion_mnist():
    """
    Load Fashion MNIST dataset and split similarly to MNIST.
    """
    print("\n\nLoading Fashion MNIST dataset...")
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
    X, y = fashion_mnist.data.to_numpy(), fashion_mnist.target.to_numpy().astype(int)
    
    # Split: 60,000 for train+val, 10,000 for test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=10000, random_state=42, stratify=y
    )
    
    # Split train_val: 50,000 for train, 10,000 for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=10000, random_state=42, stratify=y_train_val
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_fashion_mnist_experiment(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Run the same experiment on Fashion MNIST dataset.
    """
    print("\n" + "="*70)
    print("FASHION MNIST EXPERIMENT")
    print("="*70)
    
    # Train individual classifiers
    classifiers = train_individual_classifiers_mnist(X_train, y_train, X_val, y_val)
    
    # Create and evaluate ensemble
    voting_clf = create_soft_voting_ensemble(classifiers)
    ensemble_val_acc, ensemble_test_acc = evaluate_ensemble_mnist(
        voting_clf, classifiers, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    return classifiers, voting_clf, ensemble_val_acc, ensemble_test_acc


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main function to run all experiments.
    """
    print("="*70)
    print("HOMEWORK 2: ENSEMBLE LEARNING")
    print("="*70)
    
    # Part 1: MNIST
    print("\n### PART 1: MNIST DATASET ###\n")
    X_train_mnist, X_val_mnist, X_test_mnist, y_train_mnist, y_val_mnist, y_test_mnist = load_and_split_mnist()
    
    classifiers_mnist = train_individual_classifiers_mnist(
        X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist
    )
    
    voting_clf_mnist = create_soft_voting_ensemble(classifiers_mnist)
    mnist_val_acc, mnist_test_acc = evaluate_ensemble_mnist(
        voting_clf_mnist, classifiers_mnist, 
        X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist, X_test_mnist, y_test_mnist
    )
    
    # Part 2: Fashion MNIST
    print("\n\n### PART 2: FASHION MNIST DATASET ###\n")
    X_train_fashion, X_val_fashion, X_test_fashion, y_train_fashion, y_val_fashion, y_test_fashion = load_and_split_fashion_mnist()
    
    classifiers_fashion, voting_clf_fashion, fashion_val_acc, fashion_test_acc = run_fashion_mnist_experiment(
        X_train_fashion, y_train_fashion, X_val_fashion, y_val_fashion, X_test_fashion, y_test_fashion
    )
    
    # Final Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nMNIST Dataset:")
    print(f"  Ensemble Test Accuracy: {mnist_test_acc:.4f}")
    print(f"\nFashion MNIST Dataset:")
    print(f"  Ensemble Test Accuracy: {fashion_test_acc:.4f}")
    
    print("\n" + "="*70)
    print("Experiment completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
