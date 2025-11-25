"""
Homework-2: Ensemble Learning (Chapter 7)
Multiclass Classification using MNIST and Fashion MNIST datasets

Updated to include Precision, Recall, F1 Score, and Confusion Matrix visualization.
Added support for Chinese characters in plot titles and labels.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 在主程式中加入以下程式碼
from bonus_ensemble_methods import run_bonus_experiments

# ============================================================================
# 設置中文字體 (Font Configuration for Chinese)
# ============================================================================
# 嘗試使用系統中常見的中文字體。如果這些字體不存在，警告仍可能出現，
# 但通常至少有一個能成功加載，或者您需要根據您的作業系統安裝字體。
# 在 Windows/macOS 上，通常 'Arial Unicode MS', 'Microsoft JhengHei', 'SimHei' 可用。
try:
    # 嘗試設置中文字體
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
    mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False # 解決負號'-'顯示為方塊的問題
except Exception as e:
    print(f"Warning: Failed to set Chinese font. Plots might not display Chinese characters correctly. Error: {e}")

# ============================================================================
# 輔助函數：評估並顯示結果 (新增了指標和混淆矩陣)
# ============================================================================

def evaluate_and_display_metrics(y_true, y_pred, model_name, dataset_name):
    """計算並顯示多項分類指標，以及生成混淆矩陣圖表。"""
    
    # 計算 Precision, Recall, F1 Score
    # average='weighted' 表示加權平均，適用於多類別分類
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 計算單一類別的 Precision, Recall, F1 Score (用於詳細表格)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print(f"\n--- {model_name} Performance on {dataset_name} Test Set ---")
    print(f"Accuracy (準確率): {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision (精確率, Weighted): {precision:.4f}")
    print(f"Recall (召回率, Weighted): {recall:.4f}")
    print(f"F1 Score (F1 分數, Weighted): {f1:.4f}")
    
    print(f"Precision (精確率, Macro Avg): {precision_macro:.4f}")
    print(f"Recall (召回率, Macro Avg): {recall_macro:.4f}")
    print(f"F1 Score (F1 分數, Macro Avg): {f1_macro:.4f}")

    # 生成混淆矩陣
    cm = confusion_matrix(y_true, y_pred)
    
    # 獲取類別標籤 (0-9)
    class_labels = np.unique(y_true)
    
    plt.figure(figsize=(10, 8))
    # 使用 seaborn 繪製混淆矩陣熱圖
    # 這裡無需指定字體，因為我們已經全局設置了 font.family
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        xticklabels=class_labels, 
        yticklabels=class_labels
    )
    plt.title(f'{model_name} - Confusion Matrix on {dataset_name} Test Set (混淆矩陣)', fontsize=16)
    plt.ylabel('True Label (真實標籤)', fontsize=14)
    plt.xlabel('Predicted Label (預測標籤)', fontsize=14)
    plt.show()


# ============================================================================
# Part 1: The MNIST Dataset
# ============================================================================

def load_and_split_mnist():
    """
    Load MNIST dataset and split into training (50,000), validation (10,000), 
    and test (10,000) sets.
    """
    print("Loading MNIST dataset...")
    # 確保資料為 float32 以供 SVM 處理，並進行標準化
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data.to_numpy().astype(np.float32), mnist.target.to_numpy().astype(int)
    # 進行特徵標準化 (Normalization)
    X = X / 255.0
    
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
    # 增加樹的數量以提高準確性，同時 n_jobs=-1 加速訓練
    rf_clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    rf_val_acc = accuracy_score(y_val, rf_clf.predict(X_val))
    print(f"Random Forest - Validation Accuracy: {rf_val_acc:.4f} (Time: {time.time()-start:.2f}s)")
    classifiers['Random Forest'] = (rf_clf, rf_val_acc)
    
    # 2. Extra Trees
    print("\n[2/3] Training Extra Trees...")
    start = time.time()
    et_clf = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    et_clf.fit(X_train, y_train)
    et_val_acc = accuracy_score(y_val, et_clf.predict(X_val))
    print(f"Extra Trees - Validation Accuracy: {et_val_acc:.4f} (Time: {time.time()-start:.2f}s)")
    classifiers['Extra Trees'] = (et_clf, et_val_acc)
    
    # 3. SVM (Using a smaller, representative subset for SVM due to cost)
    print("\n[3/3] Training SVM...")
    # 為了加速，只使用 10000 筆訓練數據來訓練 SVM
    X_train_subset, y_train_subset = X_train[:10000], y_train[:10000]
    
    start = time.time()
    # SVM 需要 probability=True 才能用於 Soft Voting
    svm_clf = SVC(kernel='rbf', gamma='scale', random_state=42, probability=True) 
    svm_clf.fit(X_train_subset, y_train_subset)
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
    # 使用 soft voting
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    
    return voting_clf


def evaluate_ensemble_mnist(voting_clf, classifiers, X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    """
    Train and evaluate the ensemble on validation and test sets.
    """
    print(f"\nTraining Soft Voting Ensemble for {dataset_name}...")
    start = time.time()
    # 注意：VotingClassifier 會重新訓練所有內部估計器（除非它們是 prefit=True，但通常不這麼做）
    # 這裡我們使用完整的訓練集
    voting_clf.fit(X_train, y_train) 
    print(f"Training completed in {time.time()-start:.2f}s")
    
    # 取得集成模型的預測結果
    test_pred_ensemble = voting_clf.predict(X_test)
    ensemble_test_acc = accuracy_score(y_test, test_pred_ensemble)
    
    # 評估並顯示集成模型的詳細指標和混淆矩陣
    evaluate_and_display_metrics(y_test, test_pred_ensemble, "Soft Voting Ensemble", dataset_name)
    
    # 比較個體分類器的效能
    print("\n" + "="*70)
    print(f"Individual Classifier Performance on {dataset_name} Test Set (Detailed Metrics)")
    print("="*70)
    
    best_individual_test_acc = 0
    
    for name, (clf, val_acc) in classifiers.items():
        test_pred = clf.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        # 顯示個體模型的 Accuracy
        print(f"{name:20s} - Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_individual_test_acc:
            best_individual_test_acc = test_acc
            
        # 評估並顯示個體模型的詳細指標 (現在啟用，將顯示 Precision/Recall/F1/Confusion Matrix)
        evaluate_and_display_metrics(y_test, test_pred, name, dataset_name)
    
    # 顯示集成模型的 Accuracy
    print(f"{'Soft Voting Ensemble':20s} - Test Acc: {ensemble_test_acc:.4f}")
    
    # 計算提升幅度
    improvement = ensemble_test_acc - best_individual_test_acc
    print(f"\nImprovement over best individual: {improvement:.4f} ({improvement*100:.2f}%)")
    
    return ensemble_test_acc


# ============================================================================
# Part 2: The Fashion MNIST Dataset
# ============================================================================

def load_and_split_fashion_mnist():
    """
    Load Fashion MNIST dataset and split similarly to MNIST.
    """
    print("\n\nLoading Fashion MNIST dataset...")
    # 確保資料為 float32 以供 SVM 處理，並進行標準化
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, parser='auto')
    X, y = fashion_mnist.data.to_numpy().astype(np.float32), fashion_mnist.target.to_numpy().astype(int)
    # 進行特徵標準化 (Normalization)
    X = X / 255.0

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
    # 由於 Fashion MNIST 的訓練速度較慢，這裡沿用 MNIST 的訓練函數
    classifiers = train_individual_classifiers_mnist(X_train, y_train, X_val, y_val)
    
    # Create and evaluate ensemble
    voting_clf = create_soft_voting_ensemble(classifiers)
    fashion_test_acc = evaluate_ensemble_mnist(
        voting_clf, classifiers, X_train, y_train, X_val, y_val, X_test, y_test, "Fashion MNIST"
    )

    #NEW
    # 執行進階方法
    bonus_results_fashion = run_bonus_experiments(
        X_train, y_train, X_val, y_val, X_test, y_test, classifiers
    )
    
    
    return fashion_test_acc


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
    
    # 訓練個體模型
    classifiers_mnist = train_individual_classifiers_mnist(
        X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist
    )
    
    # 建立並評估集成模型
    voting_clf_mnist = create_soft_voting_ensemble(classifiers_mnist)
    mnist_test_acc = evaluate_ensemble_mnist(
        voting_clf_mnist, classifiers_mnist, 
        X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist, X_test_mnist, y_test_mnist, "MNIST"
    )

    #NEW
    # 執行進階方法
    bonus_results_mnist = run_bonus_experiments(
        X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist, X_test_mnist, y_test_mnist, classifiers_mnist
    )
    
    # Part 2: Fashion MNIST
    print("\n\n### PART 2: FASHION MNIST DATASET ###\n")
    X_train_fashion, X_val_fashion, X_test_fashion, y_train_fashion, y_val_fashion, y_test_fashion = load_and_split_fashion_mnist()
    
    # 運行 Fashion MNIST 實驗
    fashion_test_acc = run_fashion_mnist_experiment(
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