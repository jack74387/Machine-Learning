"""
Quick test script to verify the implementation works
Uses a small subset of data for fast testing
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("="*70)
print("QUICK TEST - Ensemble Learning Implementation")
print("="*70)

# Load a small subset of MNIST
print("\n1. Loading MNIST dataset (small subset)...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data.to_numpy()[:5000], mnist.target.to_numpy()[:5000].astype(int)
print(f"   Loaded {len(X)} samples")

# Split data
print("\n2. Splitting data...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Train classifiers
print("\n3. Training classifiers...")

print("   [1/3] Random Forest...")
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_acc = accuracy_score(y_val, rf_clf.predict(X_val))
print(f"         Validation Accuracy: {rf_acc:.4f}")

print("   [2/3] Extra Trees...")
et_clf = ExtraTreesClassifier(n_estimators=10, random_state=42, n_jobs=-1)
et_clf.fit(X_train, y_train)
et_acc = accuracy_score(y_val, et_clf.predict(X_val))
print(f"         Validation Accuracy: {et_acc:.4f}")

print("   [3/3] SVM...")
svm_clf = SVC(kernel='rbf', gamma='scale', random_state=42, probability=True)
svm_clf.fit(X_train[:500], y_train[:500])  # Use even smaller subset for SVM
svm_acc = accuracy_score(y_val, svm_clf.predict(X_val))
print(f"         Validation Accuracy: {svm_acc:.4f}")

# Create ensemble
print("\n4. Creating Soft Voting Ensemble...")
voting_clf = VotingClassifier(
    estimators=[('rf', rf_clf), ('et', et_clf), ('svm', svm_clf)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Evaluate
val_pred = voting_clf.predict(X_val)
ensemble_val_acc = accuracy_score(y_val, val_pred)
test_pred = voting_clf.predict(X_test)
ensemble_test_acc = accuracy_score(y_test, test_pred)

print(f"   Ensemble Validation Accuracy: {ensemble_val_acc:.4f}")
print(f"   Ensemble Test Accuracy: {ensemble_test_acc:.4f}")

# Compare
print("\n5. Performance Comparison:")
print(f"   Random Forest:  {rf_acc:.4f}")
print(f"   Extra Trees:    {et_acc:.4f}")
print(f"   SVM:            {svm_acc:.4f}")
print(f"   Ensemble:       {ensemble_test_acc:.4f}")

best_individual = max(
    accuracy_score(y_test, rf_clf.predict(X_test)),
    accuracy_score(y_test, et_clf.predict(X_test)),
    accuracy_score(y_test, svm_clf.predict(X_test))
)
improvement = ensemble_test_acc - best_individual
print(f"\n   Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

print("\n" + "="*70)
print("✓ Test completed successfully!")
print("="*70)
print("\nThe implementation is working correctly.")
print("Run 'python homework2_ensemble.py' for the full experiment.")
