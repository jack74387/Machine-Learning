"""
全面實驗：包含更多評估指標和實驗設置
- Precision, Recall, F1 Score
- 旋轉數據增強
- 全面的超參數調整
- 不同的SGD參數組合
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from scipy.ndimage import shift, rotate
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveExperiments:
    def __init__(self):
        self.results = {}
        self.detailed_results = {}
        
    def load_datasets(self):
        """載入數據集"""
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
        
    def get_detailed_metrics(self, y_true, y_pred, dataset_name, method_name):
        """獲取詳細的評估指標"""
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = np.mean(y_true == y_pred)
        
        # 每個類別的指標
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_precision': class_precision,
            'class_recall': class_recall,
            'class_f1': class_f1
        }
    
    def rotate_image(self, image, angle):
        """旋轉圖像"""
        image = image.reshape(28, 28)
        rotated = rotate(image, angle, reshape=False, cval=0)
        return rotated.reshape(784)
    
    def shift_image(self, image, dx, dy):
        """位移圖像"""
        image = image.reshape(28, 28)
        shifted = shift(image, [dy, dx], cval=0, mode='constant')
        return shifted.reshape(784)
    
    def advanced_augmentation(self, X, y, aug_size=10000, method='mixed'):
        """高級數據增強"""
        print(f"生成 {aug_size} 個增強樣本，方法: {method}")
        
        indices = np.random.choice(len(X), aug_size, replace=True)
        augmented_X = []
        augmented_y = []
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"進度: {i}/{aug_size}")
            
            original_image = X[idx]
            
            if method == 'rotation':
                # 隨機旋轉 -15 到 15 度
                angle = np.random.uniform(-15, 15)
                aug_image = self.rotate_image(original_image, angle)
            elif method == 'shift':
                # 隨機位移
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                aug_image = self.shift_image(original_image, dx, dy)
            elif method == 'mixed':
                # 混合增強
                if np.random.random() < 0.5:
                    angle = np.random.uniform(-10, 10)
                    aug_image = self.rotate_image(original_image, angle)
                else:
                    dx = np.random.randint(-1, 2)
                    dy = np.random.randint(-1, 2)
                    aug_image = self.shift_image(original_image, dx, dy)
            
            augmented_X.append(aug_image)
            augmented_y.append(y[idx])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def comprehensive_hyperparameter_tuning(self, X, y, dataset_name):
        """全面的超參數調整"""
        print(f"\n進行 {dataset_name} 全面超參數調整...")
        
        # 正規化數據
        X_normalized = X / 255.0
        
        # 定義參數網格
        param_grid = {
            'alpha': [0.1, 0.01, 0.001, 0.0001, 0.00001],
            'max_iter': [500, 1000, 1500, 2000],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.01, 0.1, 1.0],
            'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge']
        }
        
        # 網格搜索
        sgd = SGDClassifier(random_state=42)
        grid_search = GridSearchCV(sgd, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_normalized, y)
        
        print(f"最佳參數: {grid_search.best_params_}")
        print(f"最佳分數: {grid_search.best_score_:.4f}")
        
        # 使用最佳參數進行詳細評估
        best_model = grid_search.best_estimator_
        y_pred = cross_val_predict(best_model, X_normalized, y, cv=3)
        
        metrics = self.get_detailed_metrics(y, y_pred, dataset_name, 'hypertuned')
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'metrics': metrics,
            'model': best_model
        }
    
    def test_different_augmentations(self, X, y, dataset_name):
        """測試不同的數據增強方法"""
        print(f"\n測試 {dataset_name} 不同數據增強方法...")
        
        X_normalized = X / 255.0
        augmentation_results = {}
        
        methods = ['rotation', 'shift', 'mixed']
        
        for method in methods:
            print(f"\n測試增強方法: {method}")
            
            # 生成增強數據
            aug_X, aug_y = self.advanced_augmentation(X, y, 15000, method)
            aug_X_normalized = aug_X / 255.0
            
            # 合併數據
            X_combined = np.vstack([X_normalized, aug_X_normalized])
            y_combined = np.hstack([y, aug_y])
            
            # 訓練和評估
            sgd = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
            cv_scores = cross_val_score(sgd, X_combined, y_combined, cv=3, scoring='accuracy')
            
            # 獲取詳細指標
            y_pred = cross_val_predict(sgd, X_combined, y_combined, cv=3)
            metrics = self.get_detailed_metrics(y_combined, y_pred, dataset_name, method)
            
            augmentation_results[method] = {
                'cv_scores': cv_scores,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'metrics': metrics
            }
            
            print(f"{method} 準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return augmentation_results
    
    def test_preprocessing_methods(self, X, y, dataset_name):
        """測試不同的預處理方法"""
        print(f"\n測試 {dataset_name} 不同預處理方法...")
        
        preprocessing_results = {}
        
        # 1. 無預處理
        sgd_raw = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_raw = cross_val_score(sgd_raw, X, y, cv=3, scoring='accuracy')
        y_pred_raw = cross_val_predict(sgd_raw, X, y, cv=3)
        metrics_raw = self.get_detailed_metrics(y, y_pred_raw, dataset_name, 'raw')
        
        preprocessing_results['raw'] = {
            'mean_accuracy': cv_scores_raw.mean(),
            'std_accuracy': cv_scores_raw.std(),
            'metrics': metrics_raw
        }
        
        # 2. 簡單正規化 (除以255)
        X_normalized = X / 255.0
        sgd_norm = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_norm = cross_val_score(sgd_norm, X_normalized, y, cv=3, scoring='accuracy')
        y_pred_norm = cross_val_predict(sgd_norm, X_normalized, y, cv=3)
        metrics_norm = self.get_detailed_metrics(y, y_pred_norm, dataset_name, 'normalized')
        
        preprocessing_results['normalized'] = {
            'mean_accuracy': cv_scores_norm.mean(),
            'std_accuracy': cv_scores_norm.std(),
            'metrics': metrics_norm
        }
        
        # 3. 標準化 (StandardScaler)
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)
        sgd_std = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_std = cross_val_score(sgd_std, X_standardized, y, cv=3, scoring='accuracy')
        y_pred_std = cross_val_predict(sgd_std, X_standardized, y, cv=3)
        metrics_std = self.get_detailed_metrics(y, y_pred_std, dataset_name, 'standardized')
        
        preprocessing_results['standardized'] = {
            'mean_accuracy': cv_scores_std.mean(),
            'std_accuracy': cv_scores_std.std(),
            'metrics': metrics_std
        }
        
        return preprocessing_results
    
    def run_comprehensive_experiments(self):
        """運行全面實驗"""
        self.load_datasets()
        
        datasets = {
            'MNIST': (self.mnist_X_train, self.mnist_y_train, self.mnist_X_test, self.mnist_y_test),
            'Fashion_MNIST': (self.fashion_X_train, self.fashion_y_train, self.fashion_X_test, self.fashion_y_test)
        }
        
        for dataset_name, (X_train, y_train, X_test, y_test) in datasets.items():
            print(f"\n{'='*60}")
            print(f"開始 {dataset_name} 全面實驗")
            print(f"{'='*60}")
            
            self.results[dataset_name] = {}
            
            # 1. 預處理方法比較
            self.results[dataset_name]['preprocessing'] = self.test_preprocessing_methods(X_train, y_train, dataset_name)
            
            # 2. 超參數調整
            self.results[dataset_name]['hyperparameter_tuning'] = self.comprehensive_hyperparameter_tuning(X_train, y_train, dataset_name)
            
            # 3. 數據增強測試
            self.results[dataset_name]['augmentation'] = self.test_different_augmentations(X_train, y_train, dataset_name)
            
            # 4. 最終測試集評估
            best_model = self.results[dataset_name]['hyperparameter_tuning']['model']
            X_test_normalized = X_test / 255.0
            y_test_pred = best_model.predict(X_test_normalized)
            test_metrics = self.get_detailed_metrics(y_test, y_test_pred, dataset_name, 'final_test')
            
            self.results[dataset_name]['final_test'] = {
                'metrics': test_metrics,
                'classification_report': classification_report(y_test, y_test_pred)
            }
        
        return self.results
    
    def print_comprehensive_summary(self):
        """打印全面總結"""
        print("\n" + "="*80)
        print("全面實驗總結報告")
        print("="*80)
        
        for dataset_name in ['MNIST', 'Fashion_MNIST']:
            print(f"\n{dataset_name} 結果總結:")
            print("-" * 50)
            
            # 預處理比較
            print("\n1. 預處理方法比較:")
            preprocessing = self.results[dataset_name]['preprocessing']
            for method, result in preprocessing.items():
                metrics = result['metrics']
                print(f"  {method:12}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
            
            # 超參數調整結果
            print(f"\n2. 最佳超參數:")
            best_params = self.results[dataset_name]['hyperparameter_tuning']['best_params']
            best_score = self.results[dataset_name]['hyperparameter_tuning']['best_score']
            print(f"  最佳分數: {best_score:.4f}")
            print(f"  最佳參數: {best_params}")
            
            # 數據增強比較
            print(f"\n3. 數據增強方法比較:")
            augmentation = self.results[dataset_name]['augmentation']
            for method, result in augmentation.items():
                metrics = result['metrics']
                print(f"  {method:8}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
            
            # 最終測試結果
            print(f"\n4. 最終測試集結果:")
            final_metrics = self.results[dataset_name]['final_test']['metrics']
            print(f"  準確率: {final_metrics['accuracy']:.4f}")
            print(f"  精確率: {final_metrics['precision']:.4f}")
            print(f"  召回率: {final_metrics['recall']:.4f}")
            print(f"  F1分數: {final_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    experiments = ComprehensiveExperiments()
    results = experiments.run_comprehensive_experiments()
    experiments.print_comprehensive_summary()