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
    
    def shift_image(self, image, dx, dy):
        """
        移動圖像指定的像素數
        Args:
            image: 28x28的圖像數組
            dx: 水平移動像素數 (正數向右，負數向左)
            dy: 垂直移動像素數 (正數向下，負數向上)
        Returns:
            移動後的圖像
        """
        image = image.reshape(28, 28)
        shifted = shift(image, [dy, dx], cval=0, mode='constant')
        return shifted.reshape(784)
    
    def augment_dataset(self, X, y, augment_size=10000):
        """
        對數據集進行增強
        Args:
            X: 原始特徵數據
            y: 原始標籤
            augment_size: 要生成的增強樣本數量
        Returns:
            增強後的特徵和標籤
        """
        print(f"Generating {augment_size} augmented samples...")
        
        # 隨機選擇要增強的樣本
        indices = np.random.choice(len(X), augment_size, replace=True)
        
        augmented_X = []
        augmented_y = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"Progress: {i}/{augment_size}")
                
            # 隨機選擇移動方向
            dx, dy = directions[np.random.randint(0, 4)]
            
            # 生成增強樣本
            augmented_image = self.shift_image(X[idx], dx, dy)
            augmented_X.append(augmented_image)
            augmented_y.append(y[idx])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def task_1_2_data_augmentation(self):
        """
        Task 1.2: 使用數據增強來改善準確率
        """
        print("\n" + "="*50)
        print("Task 1.2: Data Augmentation")
        print("="*50)
        
        # 生成增強數據
        augmented_X, augmented_y = self.augment_dataset(self.X_train, self.y_train)
        
        # 合併原始數據和增強數據
        X_combined = np.vstack([self.X_train, augmented_X])
        y_combined = np.hstack([self.y_train, augmented_y])
        
        print(f"Original training set size: {len(self.X_train)}")
        print(f"Augmented samples: {len(augmented_X)}")
        print(f"Combined training set size: {len(X_combined)}")
        
        # 使用增強後的數據進行交叉驗證
        sgd_augmented = SGDClassifier(random_state=42)
        print("Performing 3-fold cross-validation with augmented data...")
        cv_scores_augmented = cross_val_score(sgd_augmented, X_combined, y_combined, 
                                            cv=3, scoring="accuracy")
        
        print(f"Augmented CV scores: {cv_scores_augmented}")
        print(f"Augmented mean accuracy: {cv_scores_augmented.mean():.4f}")
        print(f"Augmented standard deviation: {cv_scores_augmented.std():.4f}")
        
        # 訓練最終模型
        sgd_augmented.fit(X_combined, y_combined)
        self.sgd_clf_augmented = sgd_augmented
        
        return cv_scores_augmented.mean()
    
    def task_1_3_confusion_matrix_analysis(self):
        """
        Task 1.3: 使用混淆矩陣分析並改善效能
        """
        print("\n" + "="*50)
        print("Task 1.3: Confusion Matrix Analysis and Performance Improvement")
        print("="*50)
        
        # 使用基礎模型進行預測
        y_train_pred = cross_val_predict(self.sgd_clf, self.X_train, self.y_train, cv=3)
        
        # 生成混淆矩陣
        conf_mx = confusion_matrix(self.y_train, y_train_pred)
        print("Confusion Matrix:")
        print(conf_mx)
        
        # 計算每個類別的錯誤率
        print("\nError analysis by class:")
        for i in range(10):
            total_samples = np.sum(conf_mx[i, :])
            correct_predictions = conf_mx[i, i]
            error_rate = (total_samples - correct_predictions) / total_samples
            print(f"Class {i}: Error rate = {error_rate:.4f}")
        
        # 嘗試正規化改善
        print("\nTrying normalization...")
        X_train_normalized = self.X_train / 255.0
        
        sgd_normalized = SGDClassifier(random_state=42)
        cv_scores_normalized = cross_val_score(sgd_normalized, X_train_normalized, 
                                             self.y_train, cv=3, scoring="accuracy")
        
        print(f"Normalized CV scores: {cv_scores_normalized}")
        print(f"Normalized mean accuracy: {cv_scores_normalized.mean():.4f}")
        
        # 嘗試超參數調整
        print("\nTrying hyperparameter tuning...")
        sgd_tuned = SGDClassifier(
            alpha=0.001,  # 較小的正規化參數
            max_iter=1000,  # 更多迭代
            random_state=42
        )
        
        cv_scores_tuned = cross_val_score(sgd_tuned, X_train_normalized, 
                                        self.y_train, cv=3, scoring="accuracy")
        
        print(f"Tuned CV scores: {cv_scores_tuned}")
        print(f"Tuned mean accuracy: {cv_scores_tuned.mean():.4f}")
        
        # 保存最佳模型
        sgd_tuned.fit(X_train_normalized, self.y_train)
        self.sgd_clf_best = sgd_tuned
        self.X_train_normalized = X_train_normalized
        
        return cv_scores_tuned.mean()
    
    def evaluate_on_test_set(self):
        """在測試集上評估最佳模型"""
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        
        # 正規化測試集
        X_test_normalized = self.X_test / 255.0
        
        # 預測
        y_test_pred = self.sgd_clf_best.predict(X_test_normalized)
        
        # 計算準確率
        test_accuracy = np.mean(y_test_pred == self.y_test)
        print(f"Test set accuracy: {test_accuracy:.4f}")
        
        # 生成分類報告
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        return test_accuracy

def main():
    """主執行函數"""
    print("MNIST Classification with SGDClassifier")
    print("="*50)
    
    # 初始化分類器
    classifier = MNISTClassifier()
    
    # 載入數據
    classifier.load_mnist_data()
    
    # 執行Task 1.1
    accuracy_basic = classifier.task_1_1_basic_sgd_classification()
    print(f"\nTask 1.1 completed with accuracy: {accuracy_basic:.4f}")
    
    # 執行Task 1.2
    accuracy_augmented = classifier.task_1_2_data_augmentation()
    print(f"\nTask 1.2 completed with accuracy: {accuracy_augmented:.4f}")
    
    # 比較結果
    improvement = accuracy_augmented - accuracy_basic
    print(f"\nAccuracy improvement: {improvement:.4f} ({improvement*100:.2f}%)")

if __name__ == "__main__":
    main()