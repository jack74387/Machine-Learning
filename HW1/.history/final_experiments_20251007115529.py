"""
最終實驗：修復錯誤並獲取完整的實驗數據
包含 Precision, Recall, F1 Score 和各種實驗
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from scipy.ndimage import shift, rotate
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinalExperiments:
    def __init__(self):
        self.results = {}
        
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
        
    def get_detailed_metrics(self, y_true, y_pred):
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
    
    def create_augmented_data(self, X, y, method='rotation', size=8000):
        """創建增強數據"""
        print(f"生成 {size} 個 {method} 增強樣本...")
        
        indices = np.random.choice(len(X), size, replace=True)
        augmented_X = []
        augmented_y = []
        
        for i, idx in enumerate(indices):
            if i % 2000 == 0:
                print(f"進度: {i}/{size}")
            
            original_image = X[idx]
            
            if method == 'rotation':
                # 隨機旋轉 -15 到 15 度
                angle = np.random.uniform(-15, 15)
                aug_image = self.rotate_image(original_image, angle)
            elif method == 'rotation_small':
                # 小角度旋轉 -5 到 5 度
                angle = np.random.uniform(-5, 5)
                aug_image = self.rotate_image(original_image, angle)
            elif method == 'shift':
                # 隨機位移 -2 到 2 像素
                dx = np.random.randint(-2, 3)
                dy = np.random.randint(-2, 3)
                aug_image = self.shift_image(original_image, dx, dy)
            elif method == 'shift_small':
                # 小位移 -1 到 1 像素
                dx = np.random.randint(-1, 2)
                dy = np.random.randint(-1, 2)
                aug_image = self.shift_image(original_image, dx, dy)
            
            augmented_X.append(aug_image)
            augmented_y.append(y[idx])
        
        return np.array(augmented_X), np.array(augmented_y)
    
    def test_alpha_values(self, X, y, dataset_name):
        """測試不同Alpha值"""
        print(f"\n測試 {dataset_name} Alpha值...")
        
        X_normalized = X / 255.0
        alpha_results = {}
        
        alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
        
        for alpha in alphas:
            sgd = SGDClassifier(alpha=alpha, max_iter=1000, random_state=42)
            cv_scores = cross_val_score(sgd, X_normalized, y, cv=3, scoring='accuracy')
            y_pred = cross_val_predict(sgd, X_normalized, y, cv=3)
            metrics = self.get_detailed_metrics(y, y_pred)
            
            alpha_results[alpha] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'metrics': metrics
            }
            print(f"Alpha {alpha}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return alpha_results
    
    def test_max_iter_values(self, X, y, dataset_name):
        """測試不同Max Iter值"""
        print(f"\n測試 {dataset_name} Max Iter值...")
        
        X_normalized = X / 255.0
        iter_results = {}
        
        max_iters = [500, 1000, 1500, 2000]
        
        for max_iter in max_iters:
            sgd = SGDClassifier(alpha=0.0001, max_iter=max_iter, random_state=42)
            cv_scores = cross_val_score(sgd, X_normalized, y, cv=3, scoring='accuracy')
            y_pred = cross_val_predict(sgd, X_normalized, y, cv=3)
            metrics = self.get_detailed_metrics(y, y_pred)
            
            iter_results[max_iter] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'metrics': metrics
            }
            print(f"Max Iter {max_iter}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return iter_results
    
    def test_learning_rates(self, X, y, dataset_name):
        """測試不同Learning Rate策略"""
        print(f"\n測試 {dataset_name} Learning Rate策略...")
        
        X_normalized = X / 255.0
        lr_results = {}
        
        # 修復：為需要eta0的learning rate設置合適的eta0值
        learning_rate_configs = [
            ('constant', {'eta0': 0.01}),
            ('optimal', {}),
            ('invscaling', {'eta0': 0.01}),
            ('adaptive', {'eta0': 0.01})
        ]
        
        for lr_name, extra_params in learning_rate_configs:
            try:
                sgd = SGDClassifier(alpha=0.0001, max_iter=1000, learning_rate=lr_name, 
                                  random_state=42, **extra_params)
                cv_scores = cross_val_score(sgd, X_normalized, y, cv=3, scoring='accuracy')
                y_pred = cross_val_predict(sgd, X_normalized, y, cv=3)
                metrics = self.get_detailed_metrics(y, y_pred)
                
                lr_results[lr_name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'metrics': metrics
                }
                print(f"LR {lr_name}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"LR {lr_name}: 失敗 - {str(e)}")
                lr_results[lr_name] = {'error': str(e)}
        
        return lr_results
    
    def test_loss_functions(self, X, y, dataset_name):
        """測試不同損失函數"""
        print(f"\n測試 {dataset_name} 損失函數...")
        
        X_normalized = X / 255.0
        loss_results = {}
        
        losses = ['hinge', 'log_loss', 'modified_huber', 'squared_hinge']
        
        for loss in losses:
            try:
                sgd = SGDClassifier(alpha=0.0001, max_iter=1000, loss=loss, random_state=42)
                cv_scores = cross_val_score(sgd, X_normalized, y, cv=3, scoring='accuracy')
                y_pred = cross_val_predict(sgd, X_normalized, y, cv=3)
                metrics = self.get_detailed_metrics(y, y_pred)
                
                loss_results[loss] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'metrics': metrics
                }
                print(f"Loss {loss}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"Loss {loss}: 失敗 - {str(e)}")
                loss_results[loss] = {'error': str(e)}
        
        return loss_results
    
    def test_augmentation_methods(self, X, y, dataset_name):
        """測試數據增強方法"""
        print(f"\n測試 {dataset_name} 數據增強方法...")
        
        X_normalized = X / 255.0
        augmentation_results = {}
        
        # 基礎模型（無增強）
        sgd_base = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_base = cross_val_score(sgd_base, X_normalized, y, cv=3, scoring='accuracy')
        y_pred_base = cross_val_predict(sgd_base, X_normalized, y, cv=3)
        metrics_base = self.get_detailed_metrics(y, y_pred_base)
        
        augmentation_results['baseline'] = {
            'cv_mean': cv_scores_base.mean(),
            'cv_std': cv_scores_base.std(),
            'metrics': metrics_base
        }
        print(f"Baseline: Acc={metrics_base['accuracy']:.4f}, P={metrics_base['precision']:.4f}, R={metrics_base['recall']:.4f}, F1={metrics_base['f1_score']:.4f}")
        
        # 測試不同增強方法
        methods = ['rotation', 'rotation_small', 'shift', 'shift_small']
        
        for method in methods:
            print(f"\n測試 {method} 增強...")
            
            # 生成增強數據
            aug_X, aug_y = self.create_augmented_data(X, y, method, 8000)
            aug_X_normalized = aug_X / 255.0
            
            # 合併數據
            X_combined = np.vstack([X_normalized, aug_X_normalized])
            y_combined = np.hstack([y, aug_y])
            
            # 訓練和評估
            sgd = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
            cv_scores = cross_val_score(sgd, X_combined, y_combined, cv=3, scoring='accuracy')
            y_pred = cross_val_predict(sgd, X_combined, y_combined, cv=3)
            metrics = self.get_detailed_metrics(y_combined, y_pred)
            
            augmentation_results[method] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'metrics': metrics,
                'improvement': metrics['accuracy'] - metrics_base['accuracy']
            }
            
            print(f"{method}: Acc={metrics['accuracy']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
            print(f"改善: {metrics['accuracy'] - metrics_base['accuracy']:+.4f}")
        
        return augmentation_results
    
    def test_preprocessing_methods(self, X, y, dataset_name):
        """測試預處理方法"""
        print(f"\n測試 {dataset_name} 預處理方法...")
        
        preprocessing_results = {}
        
        # 1. 原始數據
        sgd_raw = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_raw = cross_val_score(sgd_raw, X, y, cv=3, scoring='accuracy')
        y_pred_raw = cross_val_predict(sgd_raw, X, y, cv=3)
        metrics_raw = self.get_detailed_metrics(y, y_pred_raw)
        
        preprocessing_results['raw'] = {
            'cv_mean': cv_scores_raw.mean(),
            'cv_std': cv_scores_raw.std(),
            'metrics': metrics_raw
        }
        print(f"Raw: Acc={metrics_raw['accuracy']:.4f}, P={metrics_raw['precision']:.4f}, R={metrics_raw['recall']:.4f}, F1={metrics_raw['f1_score']:.4f}")
        
        # 2. 正規化 (0-1)
        X_normalized = X / 255.0
        sgd_norm = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_norm = cross_val_score(sgd_norm, X_normalized, y, cv=3, scoring='accuracy')
        y_pred_norm = cross_val_predict(sgd_norm, X_normalized, y, cv=3)
        metrics_norm = self.get_detailed_metrics(y, y_pred_norm)
        
        preprocessing_results['normalized'] = {
            'cv_mean': cv_scores_norm.mean(),
            'cv_std': cv_scores_norm.std(),
            'metrics': metrics_norm
        }
        print(f"Normalized: Acc={metrics_norm['accuracy']:.4f}, P={metrics_norm['precision']:.4f}, R={metrics_norm['recall']:.4f}, F1={metrics_norm['f1_score']:.4f}")
        
        # 3. 標準化
        scaler = StandardScaler()
        X_standardized = scaler.fit_transform(X)
        sgd_std = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        cv_scores_std = cross_val_score(sgd_std, X_standardized, y, cv=3, scoring='accuracy')
        y_pred_std = cross_val_predict(sgd_std, X_standardized, y, cv=3)
        metrics_std = self.get_detailed_metrics(y, y_pred_std)
        
        preprocessing_results['standardized'] = {
            'cv_mean': cv_scores_std.mean(),
            'cv_std': cv_scores_std.std(),
            'metrics': metrics_std
        }
        print(f"Standardized: Acc={metrics_std['accuracy']:.4f}, P={metrics_std['precision']:.4f}, R={metrics_std['recall']:.4f}, F1={metrics_std['f1_score']:.4f}")
        
        return preprocessing_results
    
    def final_test_evaluation(self, X_train, y_train, X_test, y_test, dataset_name):
        """最終測試集評估"""
        print(f"\n{dataset_name} 最終測試集評估...")
        
        # 使用最佳配置
        X_train_norm = X_train / 255.0
        X_test_norm = X_test / 255.0
        
        # 訓練最佳模型
        best_sgd = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        best_sgd.fit(X_train_norm, y_train)
        
        # 測試集預測
        y_test_pred = best_sgd.predict(X_test_norm)
        test_metrics = self.get_detailed_metrics(y_test, y_test_pred)
        
        # 生成分類報告
        class_names = None
        if dataset_name == 'Fashion_MNIST':
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        classification_rep = classification_report(y_test, y_test_pred, target_names=class_names)
        
        return {
            'metrics': test_metrics,
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
    
    def run_all_experiments(self):
        """運行所有實驗"""
        self.load_datasets()
        
        datasets = {
            'MNIST': (self.mnist_X_train, self.mnist_y_train, self.mnist_X_test, self.mnist_y_test),
            'Fashion_MNIST': (self.fashion_X_train, self.fashion_y_train, self.fashion_X_test, self.fashion_y_test)
        }
        
        for dataset_name, (X_train, y_train, X_test, y_test) in datasets.items():
            print(f"\n{'='*60}")
            print(f"開始 {dataset_name} 完整實驗")
            print(f"{'='*60}")
            
            self.results[dataset_name] = {}
            
            # 1. 預處理方法測試
            self.results[dataset_name]['preprocessing'] = self.test_preprocessing_methods(X_train, y_train, dataset_name)
            
            # 2. Alpha值測試
            self.results[dataset_name]['alpha'] = self.test_alpha_values(X_train, y_train, dataset_name)
            
            # 3. Max Iter測試
            self.results[dataset_name]['max_iter'] = self.test_max_iter_values(X_train, y_train, dataset_name)
            
            # 4. Learning Rate測試
            self.results[dataset_name]['learning_rate'] = self.test_learning_rates(X_train, y_train, dataset_name)
            
            # 5. 損失函數測試
            self.results[dataset_name]['loss_function'] = self.test_loss_functions(X_train, y_train, dataset_name)
            
            # 6. 數據增強測試
            self.results[dataset_name]['augmentation'] = self.test_augmentation_methods(X_train, y_train, dataset_name)
            
            # 7. 最終測試
            self.results[dataset_name]['final_test'] = self.final_test_evaluation(X_train, y_train, X_test, y_test, dataset_name)
        
        return self.results
    
    def print_comprehensive_summary(self):
        """打印全面總結"""
        print("\n" + "="*80)
        print("完整實驗總結報告")
        print("="*80)
        
        for dataset_name in ['MNIST', 'Fashion_MNIST']:
            print(f"\n{dataset_name} 詳細結果:")
            print("-" * 60)
            
            # 最終測試結果
            final_metrics = self.results[dataset_name]['final_test']['metrics']
            print(f"\n最終測試集結果:")
            print(f"  準確率: {final_metrics['accuracy']:.4f}")
            print(f"  精確率: {final_metrics['precision']:.4f}")
            print(f"  召回率: {final_metrics['recall']:.4f}")
            print(f"  F1分數: {final_metrics['f1_score']:.4f}")
            
            # 最佳Alpha
            alpha_results = self.results[dataset_name]['alpha']
            best_alpha = max(alpha_results.keys(), key=lambda k: alpha_results[k]['metrics']['accuracy'])
            print(f"\n最佳Alpha: {best_alpha}")
            print(f"  準確率: {alpha_results[best_alpha]['metrics']['accuracy']:.4f}")
            
            # 最佳數據增強
            aug_results = self.results[dataset_name]['augmentation']
            best_aug = max(aug_results.keys(), key=lambda k: aug_results[k]['metrics']['accuracy'])
            print(f"\n最佳數據增強: {best_aug}")
            if 'improvement' in aug_results[best_aug]:
                print(f"  改善: {aug_results[best_aug]['improvement']:+.4f}")
            
            # 預處理比較
            print(f"\n預處理方法比較:")
            preprocessing = self.results[dataset_name]['preprocessing']
            for method, result in preprocessing.items():
                metrics = result['metrics']
                print(f"  {method:12}: {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    experiments = FinalExperiments()
    results = experiments.run_all_experiments()
    experiments.print_comprehensive_summary()