"""
擴展實驗：為報告收集更多數據
進行多次訓練和不同參數設置的實驗
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import shift
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

class ExtendedExperiments:
    def __init__(self):
        self.results = {}
        
    def load_datasets(self):
        """載入兩個數據集"""
        print("載入數據集...")
        
        # MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.mnist_X, self.mnist_y = mnist["data"], mnist["target"].astype(np.uint8)
        self.mnist_X_train, self.mnist_X_test = self.mnist_X[:60000], self.mnist_X[60000:]
        self.mnist_y_train, self.mnist_y_test = self.mnist_y[:60000], self.mnist_y[60000:]
        
        # Fashion MNIST
        (fashion_X_train, fashion_y_train), (fashion_X_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
        self.fashion_X_train = fashion_X_train.reshape(60000, 784).astype(np.float64)
        self.fashion_X_test = fashion_X_test.reshape(10000, 784).astype(np.float64)
        self.fashion_y_train = fashion_y_train.astype(np.uint8)
        self.fashion_y_test = fashion_y_test.astype(np.uint8)
        
        print("數據集載入完成")
        
    def experiment_multiple_runs(self, dataset_name, X_train, y_train, runs=5):
        """多次運行實驗獲得穩定結果"""
        print(f"\n進行{dataset_name}多次運行實驗 (共{runs}次)...")
        
        basic_scores = []
        normalized_scores = []
        tuned_scores = []
        
        for i in range(runs):
            print(f"第 {i+1}/{runs} 次運行...")
            
            # 基礎SGD
            sgd_basic = SGDClassifier(random_state=42+i)
            basic_cv = cross_val_score(sgd_basic, X_train, y_train, cv=3, scoring="accuracy")
            basic_scores.append(basic_cv.mean())
            
            # 正規化
            X_normalized = X_train / 255.0
            sgd_norm = SGDClassifier(random_state=42+i)
            norm_cv = cross_val_score(sgd_norm, X_normalized, y_train, cv=3, scoring="accuracy")
            normalized_scores.append(norm_cv.mean())
            
            # 調整參數
            sgd_tuned = SGDClassifier(alpha=0.001, max_iter=1000, random_state=42+i)
            tuned_cv = cross_val_score(sgd_tuned, X_normalized, y_train, cv=3, scoring="accuracy")
            tuned_scores.append(tuned_cv.mean())
        
        results = {
            'basic': {
                'mean': np.mean(basic_scores),
                'std': np.std(basic_scores),
                'scores': basic_scores
            },
            'normalized': {
                'mean': np.mean(normalized_scores),
                'std': np.std(normalized_scores),
                'scores': normalized_scores
            },
            'tuned': {
                'mean': np.mean(tuned_scores),
                'std': np.std(tuned_scores),
                'scores': tuned_scores
            }
        }
        
        return results
    
    def experiment_different_alphas(self, dataset_name, X_train, y_train):
        """測試不同alpha值的影響"""
        print(f"\n測試{dataset_name}不同alpha值...")
        
        X_normalized = X_train / 255.0
        alphas = [0.1, 0.01, 0.001, 0.0001]
        alpha_results = {}
        
        for alpha in alphas:
            sgd = SGDClassifier(alpha=alpha, max_iter=1000, random_state=42)
            cv_scores = cross_val_score(sgd, X_normalized, y_train, cv=3, scoring="accuracy")
            alpha_results[alpha] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            print(f"Alpha {alpha}: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return alpha_results
    
    def experiment_augmentation_sizes(self, dataset_name, X_train, y_train):
        """測試不同數據增強大小的影響"""
        print(f"\n測試{dataset_name}不同增強數據量...")
        
        def shift_image(image, dx, dy):
            image = image.reshape(28, 28)
            shifted = shift(image, [dy, dx], cval=0, mode='constant')
            return shifted.reshape(784)
        
        def augment_dataset(X, y, size):
            indices = np.random.choice(len(X), size, replace=True)
            augmented_X = []
            augmented_y = []
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for idx in indices:
                dx, dy = directions[np.random.randint(0, 4)]
                augmented_image = shift_image(X[idx], dx, dy)
                augmented_X.append(augmented_image)
                augmented_y.append(y[idx])
            
            return np.array(augmented_X), np.array(augmented_y)
        
        sizes = [5000, 10000, 20000]
        aug_results = {}
        
        for size in sizes:
            print(f"增強數據量: {size}")
            aug_X, aug_y = augment_dataset(X_train, y_train, size)
            X_combined = np.vstack([X_train, aug_X])
            y_combined = np.hstack([y_train, aug_y])
            
            sgd = SGDClassifier(random_state=42)
            cv_scores = cross_val_score(sgd, X_combined, y_combined, cv=3, scoring="accuracy")
            aug_results[size] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            print(f"準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        return aug_results
    
    def run_all_experiments(self):
        """運行所有實驗"""
        self.load_datasets()
        
        # MNIST實驗
        print("="*50)
        print("MNIST 擴展實驗")
        print("="*50)
        
        self.results['mnist'] = {}
        self.results['mnist']['multiple_runs'] = self.experiment_multiple_runs(
            'MNIST', self.mnist_X_train, self.mnist_y_train)
        self.results['mnist']['alpha_test'] = self.experiment_different_alphas(
            'MNIST', self.mnist_X_train, self.mnist_y_train)
        self.results['mnist']['augmentation'] = self.experiment_augmentation_sizes(
            'MNIST', self.mnist_X_train, self.mnist_y_train)
        
        # Fashion MNIST實驗
        print("\n" + "="*50)
        print("Fashion MNIST 擴展實驗")
        print("="*50)
        
        self.results['fashion_mnist'] = {}
        self.results['fashion_mnist']['multiple_runs'] = self.experiment_multiple_runs(
            'Fashion MNIST', self.fashion_X_train, self.fashion_y_train)
        self.results['fashion_mnist']['alpha_test'] = self.experiment_different_alphas(
            'Fashion MNIST', self.fashion_X_train, self.fashion_y_train)
        self.results['fashion_mnist']['augmentation'] = self.experiment_augmentation_sizes(
            'Fashion MNIST', self.fashion_X_train, self.fashion_y_train)
        
        return self.results
    
    def print_summary(self):
        """打印實驗總結"""
        print("\n" + "="*60)
        print("擴展實驗總結")
        print("="*60)
        
        for dataset in ['mnist', 'fashion_mnist']:
            dataset_name = 'MNIST' if dataset == 'mnist' else 'Fashion MNIST'
            print(f"\n{dataset_name} 結果:")
            
            # 多次運行結果
            mr = self.results[dataset]['multiple_runs']
            print(f"基礎SGD (5次平均): {mr['basic']['mean']:.4f} (±{mr['basic']['std']:.4f})")
            print(f"正規化 (5次平均): {mr['normalized']['mean']:.4f} (±{mr['normalized']['std']:.4f})")
            print(f"調整參數 (5次平均): {mr['tuned']['mean']:.4f} (±{mr['tuned']['std']:.4f})")
            
            # 最佳alpha
            alpha_results = self.results[dataset]['alpha_test']
            best_alpha = max(alpha_results.keys(), key=lambda k: alpha_results[k]['mean'])
            print(f"最佳Alpha: {best_alpha} (準確率: {alpha_results[best_alpha]['mean']:.4f})")

if __name__ == "__main__":
    experiments = ExtendedExperiments()
    results = experiments.run_all_experiments()
    experiments.print_summary()