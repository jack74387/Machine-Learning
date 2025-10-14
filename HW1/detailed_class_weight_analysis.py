"""
詳細的類別權重分析
包含每個類別的詳細性能分析和最佳權重搜索
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
import pandas as pd

class DetailedClassWeightAnalysis:
    def __init__(self):
        self.fashion_X_train = None
        self.fashion_y_train = None
        self.fashion_X_test = None
        self.fashion_y_test = None
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    def load_fashion_mnist(self):
        """載入Fashion MNIST數據集"""
        print("載入Fashion MNIST數據集...")
        (fashion_X_train, fashion_y_train), (fashion_X_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
        
        self.fashion_X_train = fashion_X_train.reshape(60000, 784).astype(np.float64)
        self.fashion_X_test = fashion_X_test.reshape(10000, 784).astype(np.float64)
        self.fashion_y_train = fashion_y_train.astype(np.uint8)
        self.fashion_y_test = fashion_y_test.astype(np.uint8)
        
        print("數據集載入完成")
        
    def get_per_class_metrics(self, y_true, y_pred):
        """獲取每個類別的詳細指標"""
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        class_metrics = {}
        for i in range(10):
            class_metrics[i] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        
        # 整體指標
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        overall_accuracy = np.mean(y_true == y_pred)
        
        return class_metrics, {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        }
    
    def experiment_systematic_weights(self):
        """系統性測試不同權重組合"""
        print("進行系統性權重調整實驗...")
        
        X_normalized = self.fashion_X_train / 255.0
        
        # 定義不同的權重策略
        weight_strategies = {
            '無權重': None,
            
            '輕微調整': {
                0: 1.1, 1: 1.0, 2: 1.2, 3: 1.0, 4: 1.1,
                5: 1.0, 6: 1.5, 7: 1.0, 8: 1.0, 9: 1.0
            },
            
            '中等調整': {
                0: 1.3, 1: 1.0, 2: 1.4, 3: 1.1, 4: 1.3,
                5: 1.0, 6: 2.0, 7: 1.0, 8: 1.0, 9: 1.0
            },
            
            '激進調整': {
                0: 1.5, 1: 1.0, 2: 1.8, 3: 1.2, 4: 1.6,
                5: 1.0, 6: 3.0, 7: 1.0, 8: 1.0, 9: 1.0
            },
            
            '僅針對襯衫': {
                0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,
                5: 1.0, 6: 2.5, 7: 1.0, 8: 1.0, 9: 1.0
            },
            
            '平衡困難類別': {
                0: 1.2, 1: 1.0, 2: 1.3, 3: 1.0, 4: 1.4,
                5: 1.0, 6: 2.2, 7: 1.0, 8: 1.0, 9: 1.0
            }
        }
        
        results = {}
        
        for strategy_name, weights in weight_strategies.items():
            print(f"\\n測試策略: {strategy_name}")
            
            # 創建分類器
            if weights is None:
                sgd = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
            else:
                sgd = SGDClassifier(alpha=0.0001, max_iter=1000, class_weight=weights, random_state=42)
            
            # 交叉驗證
            cv_scores = cross_val_score(sgd, X_normalized, self.fashion_y_train, cv=3, scoring='accuracy')
            y_pred = cross_val_predict(sgd, X_normalized, self.fashion_y_train, cv=3)
            
            # 獲取詳細指標
            class_metrics, overall_metrics = self.get_per_class_metrics(self.fashion_y_train, y_pred)
            
            results[strategy_name] = {
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'class_metrics': class_metrics,
                'overall_metrics': overall_metrics,
                'weights': weights
            }
            
            print(f"  交叉驗證準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"  整體F1分數: {overall_metrics['f1_score']:.4f}")
            
            # 顯示襯衫類別的改善
            shirt_recall = class_metrics[6]['recall']
            print(f"  襯衫召回率: {shirt_recall:.4f}")
        
        return results
    
    def analyze_shirt_performance(self, results):
        """專門分析襯衫類別的性能變化"""
        print("\\n" + "="*60)
        print("襯衫類別性能分析")
        print("="*60)
        
        print(f"{'策略':<15} {'襯衫精確率':<12} {'襯衫召回率':<12} {'襯衫F1':<10} {'整體準確率':<12}")
        print("-" * 65)
        
        for strategy_name, result in results.items():
            shirt_metrics = result['class_metrics'][6]
            overall_acc = result['overall_metrics']['accuracy']
            
            print(f"{strategy_name:<15} {shirt_metrics['precision']:<12.4f} {shirt_metrics['recall']:<12.4f} "
                  f"{shirt_metrics['f1_score']:<10.4f} {overall_acc:<12.4f}")
    
    def test_best_strategy_on_test_set(self, results):
        """在測試集上測試最佳策略"""
        print("\\n" + "="*60)
        print("測試集最終評估")
        print("="*60)
        
        X_train_norm = self.fashion_X_train / 255.0
        X_test_norm = self.fashion_X_test / 255.0
        
        # 選擇幾個有代表性的策略在測試集上評估
        test_strategies = ['無權重', '中等調整', '平衡困難類別']
        
        test_results = {}
        
        for strategy_name in test_strategies:
            weights = results[strategy_name]['weights']
            
            # 訓練模型
            if weights is None:
                sgd = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
            else:
                sgd = SGDClassifier(alpha=0.0001, max_iter=1000, class_weight=weights, random_state=42)
            
            sgd.fit(X_train_norm, self.fashion_y_train)
            
            # 測試集預測
            y_test_pred = sgd.predict(X_test_norm)
            
            # 獲取詳細指標
            class_metrics, overall_metrics = self.get_per_class_metrics(self.fashion_y_test, y_test_pred)
            
            test_results[strategy_name] = {
                'class_metrics': class_metrics,
                'overall_metrics': overall_metrics
            }
            
            print(f"\\n{strategy_name} - 測試集結果:")
            print(f"  整體準確率: {overall_metrics['accuracy']:.4f}")
            print(f"  整體精確率: {overall_metrics['precision']:.4f}")
            print(f"  整體召回率: {overall_metrics['recall']:.4f}")
            print(f"  整體F1分數: {overall_metrics['f1_score']:.4f}")
            print(f"  襯衫召回率: {class_metrics[6]['recall']:.4f}")
            print(f"  襯衫F1分數: {class_metrics[6]['f1_score']:.4f}")
        
        return test_results
    
    def generate_final_table(self, cv_results, test_results):
        """生成最終的對比表格"""
        print("\\n" + "="*80)
        print("類別權重調整效果完整對比表")
        print("="*80)
        
        # 交叉驗證結果表格
        print("\\n【交叉驗證結果】")
        print(f"{'權重調整':<15} {'準確率':<10} {'精確率':<10} {'召回率':<10} {'F1分數':<10} {'襯衫召回率':<12}")
        print("-" * 75)
        
        baseline_acc = cv_results['無權重']['overall_metrics']['accuracy']
        
        for strategy in ['無權重', '輕微調整', '中等調整', '激進調整', '僅針對襯衫', '平衡困難類別']:
            result = cv_results[strategy]
            overall = result['overall_metrics']
            shirt_recall = result['class_metrics'][6]['recall']
            
            print(f"{strategy:<15} {overall['accuracy']:<10.4f} {overall['precision']:<10.4f} "
                  f"{overall['recall']:<10.4f} {overall['f1_score']:<10.4f} {shirt_recall:<12.4f}")
        
        # 測試集結果表格
        print("\\n【測試集結果】")
        print(f"{'權重調整':<15} {'準確率':<10} {'精確率':<10} {'召回率':<10} {'F1分數':<10} {'襯衫召回率':<12}")
        print("-" * 75)
        
        for strategy in ['無權重', '中等調整', '平衡困難類別']:
            result = test_results[strategy]
            overall = result['overall_metrics']
            shirt_recall = result['class_metrics'][6]['recall']
            
            print(f"{strategy:<15} {overall['accuracy']:<10.4f} {overall['precision']:<10.4f} "
                  f"{overall['recall']:<10.4f} {overall['f1_score']:<10.4f} {shirt_recall:<12.4f}")
        
        # 改善分析
        print("\\n【改善分析】")
        best_strategy = '平衡困難類別'
        best_cv = cv_results[best_strategy]['overall_metrics']
        best_test = test_results[best_strategy]['overall_metrics']
        baseline_cv = cv_results['無權重']['overall_metrics']
        baseline_test = test_results['無權重']['overall_metrics']
        
        print(f"最佳策略: {best_strategy}")
        print(f"交叉驗證改善: {(best_cv['accuracy'] - baseline_cv['accuracy'])*100:+.2f}%")
        print(f"測試集改善: {(best_test['accuracy'] - baseline_test['accuracy'])*100:+.2f}%")
        print(f"襯衫召回率改善: {(cv_results[best_strategy]['class_metrics'][6]['recall'] - cv_results['無權重']['class_metrics'][6]['recall'])*100:+.2f}%")
    
    def run_complete_analysis(self):
        """運行完整分析"""
        self.load_fashion_mnist()
        
        print("開始類別權重調整的完整分析...")
        
        # 系統性權重實驗
        cv_results = self.experiment_systematic_weights()
        
        # 分析襯衫性能
        self.analyze_shirt_performance(cv_results)
        
        # 測試集評估
        test_results = self.test_best_strategy_on_test_set(cv_results)
        
        # 生成最終表格
        self.generate_final_table(cv_results, test_results)
        
        return cv_results, test_results

if __name__ == "__main__":
    analysis = DetailedClassWeightAnalysis()
    cv_results, test_results = analysis.run_complete_analysis()