"""
類別權重調整實驗
根據混淆矩陣分析結果，對容易混淆的類別進行權重調整
"""

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class ClassWeightExperiment:
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
        
    def get_detailed_metrics(self, y_true, y_pred):
        """獲取詳細評估指標"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        accuracy = np.mean(y_true == y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def analyze_confusion_matrix(self):
        """分析混淆矩陣，找出需要調整權重的類別"""
        print("分析基礎模型的混淆矩陣...")
        
        # 正規化數據
        X_normalized = self.fashion_X_train / 255.0
        
        # 訓練基礎模型
        sgd_base = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        y_pred = cross_val_predict(sgd_base, X_normalized, self.fashion_y_train, cv=3)
        
        # 生成混淆矩陣
        conf_matrix = confusion_matrix(self.fashion_y_train, y_pred)
        
        # 計算每個類別的錯誤率
        error_rates = []
        for i in range(10):
            total = np.sum(conf_matrix[i, :])
            correct = conf_matrix[i, i]
            error_rate = (total - correct) / total
            error_rates.append(error_rate)
            print(f"類別 {i} ({self.class_names[i]}): 錯誤率 = {error_rate:.4f}")
        
        return conf_matrix, error_rates
    
    def experiment_baseline(self):
        """基礎實驗（無權重調整）"""
        print("\n=== 基礎實驗（無權重調整） ===")
        
        X_normalized = self.fashion_X_train / 255.0
        
        # 基礎SGD分類器
        sgd_base = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        
        # 交叉驗證
        cv_scores = cross_val_score(sgd_base, X_normalized, self.fashion_y_train, cv=3, scoring='accuracy')
        y_pred = cross_val_predict(sgd_base, X_normalized, self.fashion_y_train, cv=3)
        
        # 獲取詳細指標
        metrics = self.get_detailed_metrics(self.fashion_y_train, y_pred)
        
        print(f"交叉驗證平均準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"精確率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分數: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def experiment_with_class_weights(self):
        """使用類別權重調整的實驗"""
        print("\n=== 類別權重調整實驗 ===")
        
        X_normalized = self.fashion_X_train / 255.0
        
        # 根據混淆矩陣分析結果設計權重
        # 襯衫(6)錯誤率最高，給予最高權重
        # T恤(0)、套頭衫(2)、外套(4)也經常混淆，給予適中權重
        class_weights = {
            0: 1.3,  # T-shirt/top - 經常與襯衫混淆
            1: 1.0,  # Trouser - 表現良好
            2: 1.4,  # Pullover - 經常與襯衫、外套混淆
            3: 1.1,  # Dress - 表現中等
            4: 1.3,  # Coat - 經常與套頭衫混淆
            5: 1.0,  # Sandal - 表現良好
            6: 2.0,  # Shirt - 錯誤率最高，給予最高權重
            7: 1.0,  # Sneaker - 表現良好
            8: 1.0,  # Bag - 表現良好
            9: 1.0   # Ankle boot - 表現良好
        }
        
        print("使用的類別權重:")
        for i, weight in class_weights.items():
            print(f"  {self.class_names[i]}: {weight}")
        
        # 帶權重的SGD分類器
        sgd_weighted = SGDClassifier(
            alpha=0.0001, 
            max_iter=1000, 
            class_weight=class_weights,
            random_state=42
        )
        
        # 交叉驗證
        cv_scores = cross_val_score(sgd_weighted, X_normalized, self.fashion_y_train, cv=3, scoring='accuracy')
        y_pred = cross_val_predict(sgd_weighted, X_normalized, self.fashion_y_train, cv=3)
        
        # 獲取詳細指標
        metrics = self.get_detailed_metrics(self.fashion_y_train, y_pred)
        
        print(f"交叉驗證平均準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"精確率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分數: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def experiment_balanced_weights(self):
        """使用sklearn自動平衡權重的實驗"""
        print("\n=== 自動平衡權重實驗 ===")
        
        X_normalized = self.fashion_X_train / 255.0
        
        # 使用'balanced'自動計算權重
        sgd_balanced = SGDClassifier(
            alpha=0.0001, 
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        )
        
        # 交叉驗證
        cv_scores = cross_val_score(sgd_balanced, X_normalized, self.fashion_y_train, cv=3, scoring='accuracy')
        y_pred = cross_val_predict(sgd_balanced, X_normalized, self.fashion_y_train, cv=3)
        
        # 獲取詳細指標
        metrics = self.get_detailed_metrics(self.fashion_y_train, y_pred)
        
        print(f"交叉驗證平均準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"精確率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分數: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def experiment_aggressive_weights(self):
        """使用更激進權重調整的實驗"""
        print("\n=== 激進權重調整實驗 ===")
        
        X_normalized = self.fashion_X_train / 255.0
        
        # 更激進的權重設置，特別針對最難分類的類別
        aggressive_weights = {
            0: 1.5,  # T-shirt/top
            1: 1.0,  # Trouser
            2: 1.8,  # Pullover - 提高權重
            3: 1.2,  # Dress
            4: 1.6,  # Coat - 提高權重
            5: 1.0,  # Sandal
            6: 3.0,  # Shirt - 大幅提高權重
            7: 1.0,  # Sneaker
            8: 1.0,  # Bag
            9: 1.0   # Ankle boot
        }
        
        print("使用的激進權重:")
        for i, weight in aggressive_weights.items():
            print(f"  {self.class_names[i]}: {weight}")
        
        sgd_aggressive = SGDClassifier(
            alpha=0.0001, 
            max_iter=1000, 
            class_weight=aggressive_weights,
            random_state=42
        )
        
        # 交叉驗證
        cv_scores = cross_val_score(sgd_aggressive, X_normalized, self.fashion_y_train, cv=3, scoring='accuracy')
        y_pred = cross_val_predict(sgd_aggressive, X_normalized, self.fashion_y_train, cv=3)
        
        # 獲取詳細指標
        metrics = self.get_detailed_metrics(self.fashion_y_train, y_pred)
        
        print(f"交叉驗證平均準確率: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"精確率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分數: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def test_final_models(self):
        """在測試集上評估最佳模型"""
        print("\n=== 測試集最終評估 ===")
        
        X_train_norm = self.fashion_X_train / 255.0
        X_test_norm = self.fashion_X_test / 255.0
        
        # 測試不同的權重設置
        models = {
            '無權重': SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42),
            '手動權重': SGDClassifier(alpha=0.0001, max_iter=1000, 
                                  class_weight={0: 1.3, 1: 1.0, 2: 1.4, 3: 1.1, 4: 1.3, 
                                              5: 1.0, 6: 2.0, 7: 1.0, 8: 1.0, 9: 1.0}, 
                                  random_state=42),
            '自動平衡': SGDClassifier(alpha=0.0001, max_iter=1000, class_weight='balanced', random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # 訓練模型
            model.fit(X_train_norm, self.fashion_y_train)
            
            # 測試集預測
            y_test_pred = model.predict(X_test_norm)
            
            # 計算指標
            metrics = self.get_detailed_metrics(self.fashion_y_test, y_test_pred)
            results[name] = metrics
            
            print(f"\n{name}模型測試集結果:")
            print(f"  準確率: {metrics['accuracy']:.4f}")
            print(f"  精確率: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分數: {metrics['f1_score']:.4f}")
        
        return results
    
    def run_all_experiments(self):
        """運行所有實驗"""
        self.load_fashion_mnist()
        
        # 分析混淆矩陣
        conf_matrix, error_rates = self.analyze_confusion_matrix()
        
        # 運行各種實驗
        print("\n" + "="*60)
        print("類別權重調整實驗結果")
        print("="*60)
        
        baseline_metrics = self.experiment_baseline()
        weighted_metrics = self.experiment_with_class_weights()
        balanced_metrics = self.experiment_balanced_weights()
        aggressive_metrics = self.experiment_aggressive_weights()
        
        # 測試集評估
        test_results = self.test_final_models()
        
        # 生成結果表格
        self.generate_results_table(baseline_metrics, weighted_metrics, balanced_metrics, aggressive_metrics)
        
        return {
            'baseline': baseline_metrics,
            'weighted': weighted_metrics,
            'balanced': balanced_metrics,
            'aggressive': aggressive_metrics,
            'test_results': test_results
        }
    
    def generate_results_table(self, baseline, weighted, balanced, aggressive):
        """生成結果對比表格"""
        print("\n" + "="*80)
        print("類別權重調整效果對比表")
        print("="*80)
        
        print(f"{'權重調整':<15} {'交叉驗證準確率':<15} {'精確率':<10} {'召回率':<10} {'F1分數':<10}")
        print("-" * 70)
        
        print(f"{'無':<15} {baseline['accuracy']:.4f}          {baseline['precision']:.4f}   {baseline['recall']:.4f}   {baseline['f1_score']:.4f}")
        print(f"{'手動權重':<15} {weighted['accuracy']:.4f}          {weighted['precision']:.4f}   {weighted['recall']:.4f}   {weighted['f1_score']:.4f}")
        print(f"{'自動平衡':<15} {balanced['accuracy']:.4f}          {balanced['precision']:.4f}   {balanced['recall']:.4f}   {balanced['f1_score']:.4f}")
        print(f"{'激進權重':<15} {aggressive['accuracy']:.4f}          {aggressive['precision']:.4f}   {aggressive['recall']:.4f}   {aggressive['f1_score']:.4f}")
        
        # 計算改善幅度
        print(f"\n改善幅度 (相對於基礎模型):")
        print(f"手動權重: {(weighted['accuracy'] - baseline['accuracy'])*100:+.2f}%")
        print(f"自動平衡: {(balanced['accuracy'] - baseline['accuracy'])*100:+.2f}%")
        print(f"激進權重: {(aggressive['accuracy'] - baseline['accuracy'])*100:+.2f}%")

if __name__ == "__main__":
    experiment = ClassWeightExperiment()
    results = experiment.run_all_experiments()