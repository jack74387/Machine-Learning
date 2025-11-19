"""
Bonus: Advanced Ensemble Learning Methods
Implementing Stacking, Weighted Voting, and Gradient Boosting
"""

import numpy as np
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time


def create_stacking_ensemble(X_train, y_train, X_val, y_val):
    """
    Create a Stacking ensemble with a meta-learner.
    Base learners: Random Forest, Extra Trees, SVM
    Meta-learner: Logistic Regression
    """
    print("\n" + "="*70)
    print("Creating Stacking Ensemble")
    print("="*70)
    
    # Define base learners
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ('svm', SVC(kernel='rbf', gamma='scale', random_state=42, probability=True))
    ]
    
    # Define meta-learner
    meta_learner = LogisticRegression(max_iter=1000, random_state=42)
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,  # 5-fold cross-validation for generating meta-features
        n_jobs=-1
    )
    
    print("\nTraining Stacking Ensemble...")
    start = time.time()
    # Use subset for faster training
    stacking_clf.fit(X_train[:20000], y_train[:20000])
    print(f"Training completed in {time.time()-start:.2f}s")
    
    # Evaluate
    val_pred = stacking_clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Stacking Ensemble - Validation Accuracy: {val_acc:.4f}")
    
    return stacking_clf, val_acc


def create_weighted_voting_ensemble(classifiers, X_val, y_val):
    """
    Create a weighted voting ensemble based on validation performance.
    Better performing classifiers get higher weights.
    """
    print("\n" + "="*70)
    print("Creating Weighted Voting Ensemble")
    print("="*70)
    
    # Calculate weights based on validation accuracy
    weights = []
    for name, (clf, val_acc) in classifiers.items():
        weights.append(val_acc)
        print(f"{name}: weight = {val_acc:.4f}")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"\nNormalized weights: {weights}")
    
    return weights


def weighted_voting_predict(classifiers, weights, X):
    """
    Make predictions using weighted voting.
    """
    predictions = []
    for name, (clf, _) in classifiers.items():
        pred_proba = clf.predict_proba(X)
        predictions.append(pred_proba)
    
    # Weighted average of probabilities
    weighted_proba = np.average(predictions, axis=0, weights=weights)
    final_pred = np.argmax(weighted_proba, axis=1)
    
    return final_pred


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """
    Train a Gradient Boosting classifier.
    """
    print("\n" + "="*70)
    print("Training Gradient Boosting Classifier")
    print("="*70)
    
    print("\nTraining Gradient Boosting...")
    start = time.time()
    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    # Use subset for faster training
    gb_clf.fit(X_train[:20000], y_train[:20000])
    print(f"Training completed in {time.time()-start:.2f}s")
    
    # Evaluate
    val_pred = gb_clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f"Gradient Boosting - Validation Accuracy: {val_acc:.4f}")
    
    return gb_clf, val_acc


def run_bonus_experiments(X_train, y_train, X_val, y_val, X_test, y_test, classifiers):
    """
    Run all bonus ensemble experiments.
    """
    print("\n\n" + "="*70)
    print("BONUS: ADVANCED ENSEMBLE METHODS")
    print("="*70)
    
    results = {}
    
    # 1. Stacking Ensemble
    try:
        stacking_clf, stacking_val_acc = create_stacking_ensemble(
            X_train, y_train, X_val, y_val
        )
        stacking_test_acc = accuracy_score(y_test, stacking_clf.predict(X_test))
        results['Stacking'] = (stacking_test_acc, stacking_val_acc)
        print(f"Stacking - Test Accuracy: {stacking_test_acc:.4f}")
    except Exception as e:
        print(f"Stacking failed: {e}")
    
    # 2. Weighted Voting
    try:
        weights = create_weighted_voting_ensemble(classifiers, X_val, y_val)
        weighted_val_pred = weighted_voting_predict(classifiers, weights, X_val)
        weighted_val_acc = accuracy_score(y_val, weighted_val_pred)
        weighted_test_pred = weighted_voting_predict(classifiers, weights, X_test)
        weighted_test_acc = accuracy_score(y_test, weighted_test_pred)
        results['Weighted Voting'] = (weighted_test_acc, weighted_val_acc)
        print(f"\nWeighted Voting - Validation Accuracy: {weighted_val_acc:.4f}")
        print(f"Weighted Voting - Test Accuracy: {weighted_test_acc:.4f}")
    except Exception as e:
        print(f"Weighted Voting failed: {e}")
    
    # 3. Gradient Boosting
    try:
        gb_clf, gb_val_acc = train_gradient_boosting(X_train, y_train, X_val, y_val)
        gb_test_acc = accuracy_score(y_test, gb_clf.predict(X_test))
        results['Gradient Boosting'] = (gb_test_acc, gb_val_acc)
        print(f"Gradient Boosting - Test Accuracy: {gb_test_acc:.4f}")
    except Exception as e:
        print(f"Gradient Boosting failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("BONUS METHODS SUMMARY")
    print("="*70)
    for method, (test_acc, val_acc) in results.items():
        print(f"{method:25s} - Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    
    return results


# Example usage (to be integrated with main script)
if __name__ == "__main__":
    print("This module provides bonus ensemble methods.")
    print("Import and use with the main homework script.")
    print("\nAvailable methods:")
    print("  1. create_stacking_ensemble()")
    print("  2. create_weighted_voting_ensemble()")
    print("  3. train_gradient_boosting()")
    print("  4. run_bonus_experiments()")
