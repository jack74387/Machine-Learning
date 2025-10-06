"""
Homework-1: Multiclass Classification
Part 2: Fashion MNIST Dataset Classification using SGDClassifier

Author: Kiro AI Assistant
Date: 2025-10-06
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import shift
import warnings
warnings.filterwarnings('ignore')

class FashionMNISTClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.sgd_clf = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    def load_fashion_mnist_data(self):
        """載入Fashion MNIST數據集"""
        print("Loading Fashion MNIST dataset...")
        
        try:
            # 嘗試使用sklearn載入
            from sklearn.datasets import fetch_openml
            fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
            X, y = fashion_mnist["data"], fashion_mnist["target"].astype(np.uint8)
            
            # 分割訓練集和測試集
            self.X_train, self.X_test = X[:60000], X[60000:]
            self.y_train, self.y_test = y[:60000], y[60000:]
            
        except Exception as e:
            print(f"Error loading from sklearn: {e}")
            print("Trying alternative method...")
            
            # 備用方法：使用tensorflow/keras載入
            try:
                import tensorflow as tf
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
                
                # 展平圖像為784維向量
                self.X_train = X_train.reshape(60000, 784).astype(np.float64)
                self.X_test = X_test.reshape(10000, 784).astype(np.float64)
                self.y_train = y_train.astype(np.uint8)
                self.y_test = y_test.astype(np.uint8)
                
            except ImportError:
                print("TensorFlow not available. Please install it: pip install tensorflow")
                return False
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        print(f"Class names: {self.class_names}")
        
        return True
        
    def shift_image(self, image, dx, dy):
        """移動圖像指定的像素數"""
        image = image.reshape(28, 28)
        shifted = shift(image, [dy, dx], cval=0, mode='constant')
        return shifted.reshape(784)
    
    def augment_dataset(self, X, y, augment_size=10000):
        """對數據集進行增強"""
        print(f"Generating {augment_size} augmented samples...")
        
        indices = np.random.choice(len(X), augment_size, replace=True)
        augmented_X = []
        augmented_y = []
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"Progress: {i}/{augment_size}")
                
            dx, dy = directions[np.random.randint(0, 4)]
            augmented_image = self.shift_image(X[idx], dx, dy)
            augmented_X.append(augmented_image)
            augmented_y.append(y[idx])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def task_2_1_basic_classification(self):
        """Task 2.1: 基礎SGD分類"""
        print("\n" + "="*50)
        print("Task 2.1: Basic SGD Classification on Fashion MNIST")
        print("="*50)
        
        self.sgd_clf = SGDClassifier(random_state=42)
        
        print("Performing 3-fold cross-validation...")
        cv_scores = cross_val_score(self.sgd_clf, self.X_train, self.y_train, 
                                   cv=3, scoring="accuracy")
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean accuracy: {cv_scores.mean():.4f}")
        print(f"Standard deviation: {cv_scores.std():.4f}")
        
        self.sgd_clf.fit(self.X_train, self.y_train)
        return cv_scores.mean()
    
    def task_2_2_data_augmentation(self):
        """Task 2.2: 數據增強"""
        print("\n" + "="*50)
        print("Task 2.2: Data Augmentation on Fashion MNIST")
        print("="*50)
        
        augmented_X, augmented_y = self.augment_dataset(self.X_train, self.y_train)
        
        X_combined = np.vstack([self.X_train, augmented_X])
        y_combined = np.hstack([self.y_train, augmented_y])
        
        print(f"Original training set size: {len(self.X_train)}")
        print(f"Augmented samples: {len(augmented_X)}")
        print(f"Combined training set size: {len(X_combined)}")
        
        sgd_augmented = SGDClassifier(random_state=42)
        cv_scores_augmented = cross_val_score(sgd_augmented, X_combined, y_combined, 
                                            cv=3, scoring="accuracy")
        
        print(f"Augmented CV scores: {cv_scores_augmented}")
        print(f"Augmented mean accuracy: {cv_scores_augmented.mean():.4f}")
        
        sgd_augmented.fit(X_combined, y_combined)
        self.sgd_clf_augmented = sgd_augmented
        
        return cv_scores_augmented.mean()
    
    def task_2_3_optimization(self):
        """Task 2.3: 混淆矩陣分析與優化"""
        print("\n" + "="*50)
        print("Task 2.3: Confusion Matrix Analysis and Optimization")
        print("="*50)
        
        # 混淆矩陣分析
        y_train_pred = cross_val_predict(self.sgd_clf, self.X_train, self.y_train, cv=3)
        conf_mx = confusion_matrix(self.y_train, y_train_pred)
        
        print("Confusion Matrix:")
        print(conf_mx)
        
        print("\nError analysis by class:")
        for i in range(10):
            total_samples = np.sum(conf_mx[i, :])
            correct_predictions = conf_mx[i, i]
            error_rate = (total_samples - correct_predictions) / total_samples
            print(f"Class {i} ({self.class_names[i]}): Error rate = {error_rate:.4f}")
        
        # 正規化
        print("\nTrying normalization...")
        X_train_normalized = self.X_train / 255.0
        
        sgd_normalized = SGDClassifier(random_state=42)
        cv_scores_normalized = cross_val_score(sgd_normalized, X_train_normalized, 
                                             self.y_train, cv=3, scoring="accuracy")
        
        print(f"Normalized mean accuracy: {cv_scores_normalized.mean():.4f}")
        
        # 超參數調整
        print("\nTrying hyperparameter tuning...")
        sgd_tuned = SGDClassifier(
            alpha=0.001,
            max_iter=1000,
            random_state=42
        )
        
        cv_scores_tuned = cross_val_score(sgd_tuned, X_train_normalized, 
                                        self.y_train, cv=3, scoring="accuracy")
        
        print(f"Tuned mean accuracy: {cv_scores_tuned.mean():.4f}")
        
        sgd_tuned.fit(X_train_normalized, self.y_train)
        self.sgd_clf_best = sgd_tuned
        self.X_train_normalized = X_train_normalized
        
        return cv_scores_tuned.mean()
    
    def evaluate_on_test_set(self):
        """測試集評估"""
        print("\n" + "="*50)
        print("Final Evaluation on Fashion MNIST Test Set")
        print("="*50)
        
        X_test_normalized = self.X_test / 255.0
        y_test_pred = self.sgd_clf_best.predict(X_test_normalized)
        
        test_accuracy = np.mean(y_test_pred == self.y_test)
        print(f"Test set accuracy: {test_accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=self.class_names))
        
        return test_accuracy

def main():
    """主執行函數"""
    print("Fashion MNIST Classification with SGDClassifier")
    print("="*50)
    
    classifier = FashionMNISTClassifier()
    
    # 載入數據
    if not classifier.load_fashion_mnist_data():
        print("Failed to load Fashion MNIST dataset")
        return
    
    # 執行所有任務
    accuracy_basic = classifier.task_2_1_basic_classification()
    accuracy_augmented = classifier.task_2_2_data_augmentation()
    accuracy_optimized = classifier.task_2_3_optimization()
    test_accuracy = classifier.evaluate_on_test_set()
    
    # 總結結果
    print("\n" + "="*50)
    print("PART 2 SUMMARY - Fashion MNIST")
    print("="*50)
    print(f"Basic SGD accuracy: {accuracy_basic:.4f}")
    print(f"With data augmentation: {accuracy_augmented:.4f}")
    print(f"With optimization: {accuracy_optimized:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()