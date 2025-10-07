"""
Homework-1: Multiclass Classification
Part 1: MNIST Dataset Classification using SGDClassifier

Author: Kiro AI Assistant
Date: 2025-10-06
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import shift
import warnings
warnings.filterwarnings('ignore')

class MNISTClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.sgd_clf = None
        
    def load_mnist_data(self):
        """載入MNIST數據集"""
        print("Loading MNIST dataset...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist["data"], mnist["target"].astype(np.uint8)
        
        # 分割訓練集和測試集 (前60000為訓練，後10000為測試)
        self.X_train, self.X_test = X[:60000], X[60000:]
        self.y_train, self.y_test = y[:60000], y[60000:]
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
    def task_1_1_basic_sgd_classification(self):
        """
        Task 1.1: 使用SGDClassifier進行基礎分類，用3折交叉驗證測量準確率
        """
        print("\n" + "="*50)
        print("Task 1.1: Basic SGD Classification with Cross-Validation")
        print("="*50)
        
        # 初始化SGD分類器
        self.sgd_clf = SGDClassifier(random_state=42)
        
        # 執行3折交叉驗證
        print("Performing 3-fold cross-validation...")
        cv_scores = cross_val_score(self.sgd_clf, self.X_train, self.y_train, 
                                   cv=3, scoring="accuracy")
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean accuracy: {cv_scores.mean():.4f}")
        print(f"Standard deviation: {cv_scores.std():.4f}")
        
        # 訓練完整模型用於後續任務
        print("Training final model on full training set...")
        self.sgd_clf.fit(self.X_train, self.y_train)
        
        return cv_scores.mean()

def main():
    """主執行函數"""
    print("MNIST Classification with SGDClassifier")
    print("="*50)
    
    # 初始化分類器
    classifier = MNISTClassifier()
    
    # 載入數據
    classifier.load_mnist_data()
    
    # 執行Task 1.1
    accuracy = classifier.task_1_1_basic_sgd_classification()
    
    print(f"\nTask 1.1 completed with accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()