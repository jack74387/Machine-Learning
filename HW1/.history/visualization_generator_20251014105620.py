"""
生成混淆矩陣和錯誤分析的可視化圖片
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from matplotlib import rcParams

# 設置中文字體
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

class VisualizationGenerator:
    def __init__(self):
        self.mnist_X = None
        self.mnist_y = None
        self.fashion_X = None
        self.fashion_y = None
        
    def load_datasets(self):
        """載入數據集"""
        print("載入數據集...")
        
        # MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        self.mnist_X, self.mnist_y = mnist["data"], mnist["target"].astype(np.uint8)
        self.mnist_X_train = self.mnist_X[:60000]
        self.mnist_y_train = self.mnist_y[:60000]
        
        # Fashion MNIST
        (fashion_X_train, fashion_y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
        self.fashion_X_train = fashion_X_train.reshape(60000, 784).astype(np.float64)
        self.fashion_y_train = fashion_y_train.astype(np.uint8)
        
        print("數據集載入完成")
        
    def generate_confusion_matrices(self):
        """生成混淆矩陣"""
        print("生成混淆矩陣...")
        
        # MNIST混淆矩陣
        X_mnist_norm = self.mnist_X_train / 255.0
        sgd_mnist = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        y_mnist_pred = cross_val_predict(sgd_mnist, X_mnist_norm, self.mnist_y_train, cv=3)
        conf_mx_mnist = confusion_matrix(self.mnist_y_train, y_mnist_pred)
        
        # Fashion MNIST混淆矩陣
        X_fashion_norm = self.fashion_X_train / 255.0
        sgd_fashion = SGDClassifier(alpha=0.0001, max_iter=1000, random_state=42)
        y_fashion_pred = cross_val_predict(sgd_fashion, X_fashion_norm, self.fashion_y_train, cv=3)
        conf_mx_fashion = confusion_matrix(self.fashion_y_train, y_fashion_pred)
        
        # 繪製MNIST混淆矩陣
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.heatmap(conf_mx_mnist, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=range(10), yticklabels=range(10))
        plt.title('MNIST 混淆矩陣', fontsize=14, fontweight='bold')
        plt.xlabel('預測標籤', fontsize=12)
        plt.ylabel('真實標籤', fontsize=12)
        
        # 繪製Fashion MNIST混淆矩陣
        plt.subplot(1, 2, 2)
        fashion_labels = ['T恤', '褲子', '套頭衫', '連衣裙', '外套', 
                         '涼鞋', '襯衫', '運動鞋', '包', '靴子']
        sns.heatmap(conf_mx_fashion, annot=True, fmt='d', cmap='Reds',
                    xticklabels=fashion_labels, yticklabels=fashion_labels)
        plt.title('Fashion MNIST 混淆矩陣', fontsize=14, fontweight='bold')
        plt.xlabel('預測標籤', fontsize=12)
        plt.ylabel('真實標籤', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return conf_mx_mnist, conf_mx_fashion
    
    def find_confused_pairs(self, conf_matrix, dataset_name):
        """找出最容易混淆的類別對"""
        print(f"分析{dataset_name}混淆情況...")
        
        confused_pairs = []
        n_classes = conf_matrix.shape[0]
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and conf_matrix[i, j] > 0:
                    # 計算混淆率
                    confusion_rate = conf_matrix[i, j] / np.sum(conf_matrix[i, :])
                    confused_pairs.append((i, j, conf_matrix[i, j], confusion_rate))
        
        # 按混淆數量排序
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"{dataset_name} 前10個最容易混淆的類別對:")
        for i, (true_label, pred_label, count, rate) in enumerate(confused_pairs[:10]):
            print(f"{i+1}. 真實:{true_label} -> 預測:{pred_label}, 數量:{count}, 比率:{rate:.3f}")
        
        return confused_pairs[:10]
    
    def visualize_confused_digits(self, confused_pairs):
        """可視化容易混淆的MNIST數字樣本"""
        print("生成混淆數字樣本圖...")
        
        # 選擇前3個最容易混淆的對
        top_pairs = confused_pairs[:3]
        
        fig, axes = plt.subplots(3, 10, figsize=(15, 6))
        fig.suptitle('MNIST 容易混淆的數字對樣本', fontsize=16, fontweight='bold')
        
        for pair_idx, (true_label, pred_label, _, _) in enumerate(top_pairs):
            # 找到真實標籤的樣本
            true_indices = np.where(self.mnist_y_train == true_label)[0]
            pred_indices = np.where(self.mnist_y_train == pred_label)[0]
            
            # 隨機選擇5個樣本
            true_samples = np.random.choice(true_indices, 5, replace=False)
            pred_samples = np.random.choice(pred_indices, 5, replace=False)
            
            # 繪製真實標籤樣本
            for i, idx in enumerate(true_samples):
                img = self.mnist_X_train[idx].reshape(28, 28)
                axes[pair_idx, i].imshow(img, cmap='gray')
                axes[pair_idx, i].set_title(f'真實: {true_label}', fontsize=10)
                axes[pair_idx, i].axis('off')
            
            # 繪製預測標籤樣本
            for i, idx in enumerate(pred_samples):
                img = self.mnist_X_train[idx].reshape(28, 28)
                axes[pair_idx, i+5].imshow(img, cmap='gray')
                axes[pair_idx, i+5].set_title(f'真實: {pred_label}', fontsize=10)
                axes[pair_idx, i+5].axis('off')
        
        plt.tight_layout()
        plt.savefig('confused_mnist_digits.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_confused_fashion(self, confused_pairs):
        """可視化容易混淆的Fashion MNIST樣本"""
        print("生成混淆服裝樣本圖...")
        
        fashion_labels = ['T恤', '褲子', '套頭衫', '連衣裙', '外套', 
                         '涼鞋', '襯衫', '運動鞋', '包', '靴子']
        
        # 選擇前3個最容易混淆的對
        top_pairs = confused_pairs[:3]
        
        fig, axes = plt.subplots(3, 10, figsize=(15, 6))
        fig.suptitle('Fashion MNIST 容易混淆的服裝類別樣本', fontsize=16, fontweight='bold')
        
        for pair_idx, (true_label, pred_label, _, _) in enumerate(top_pairs):
            # 找到真實標籤的樣本
            true_indices = np.where(self.fashion_y_train == true_label)[0]
            pred_indices = np.where(self.fashion_y_train == pred_label)[0]
            
            # 隨機選擇5個樣本
            true_samples = np.random.choice(true_indices, 5, replace=False)
            pred_samples = np.random.choice(pred_indices, 5, replace=False)
            
            # 繪製真實標籤樣本
            for i, idx in enumerate(true_samples):
                img = self.fashion_X_train[idx].reshape(28, 28)
                axes[pair_idx, i].imshow(img, cmap='gray')
                axes[pair_idx, i].set_title(f'{fashion_labels[true_label]}', fontsize=10)
                axes[pair_idx, i].axis('off')
            
            # 繪製預測標籤樣本
            for i, idx in enumerate(pred_samples):
                img = self.fashion_X_train[idx].reshape(28, 28)
                axes[pair_idx, i+5].imshow(img, cmap='gray')
                axes[pair_idx, i+5].set_title(f'{fashion_labels[pred_label]}', fontsize=10)
                axes[pair_idx, i+5].axis('off')
        
        plt.tight_layout()
        plt.savefig('confused_fashion_items.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_error_analysis_chart(self, conf_mx_mnist, conf_mx_fashion):
        """生成錯誤率分析圖表"""
        print("生成錯誤率分析圖表...")
        
        # 計算每個類別的錯誤率
        mnist_error_rates = []
        fashion_error_rates = []
        
        for i in range(10):
            # MNIST錯誤率
            total_mnist = np.sum(conf_mx_mnist[i, :])
            correct_mnist = conf_mx_mnist[i, i]
            error_rate_mnist = (total_mnist - correct_mnist) / total_mnist
            mnist_error_rates.append(error_rate_mnist)
            
            # Fashion MNIST錯誤率
            total_fashion = np.sum(conf_mx_fashion[i, :])
            correct_fashion = conf_mx_fashion[i, i]
            error_rate_fashion = (total_fashion - correct_fashion) / total_fashion
            fashion_error_rates.append(error_rate_fashion)
        
        # 繪製錯誤率比較圖
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MNIST錯誤率
        bars1 = ax1.bar(range(10), mnist_error_rates, color='skyblue', alpha=0.7)
        ax1.set_title('MNIST 各數字錯誤率', fontsize=14, fontweight='bold')
        ax1.set_xlabel('數字類別', fontsize=12)
        ax1.set_ylabel('錯誤率', fontsize=12)
        ax1.set_xticks(range(10))
        ax1.set_xticklabels(range(10))
        ax1.grid(axis='y', alpha=0.3)
        
        # 在柱狀圖上添加數值
        for bar, rate in zip(bars1, mnist_error_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Fashion MNIST錯誤率
        fashion_labels = ['T恤', '褲子', '套頭衫', '連衣裙', '外套', 
                         '涼鞋', '襯衫', '運動鞋', '包', '靴子']
        bars2 = ax2.bar(range(10), fashion_error_rates, color='lightcoral', alpha=0.7)
        ax2.set_title('Fashion MNIST 各類別錯誤率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('服裝類別', fontsize=12)
        ax2.set_ylabel('錯誤率', fontsize=12)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(fashion_labels, rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # 在柱狀圖上添加數值
        for bar, rate in zip(bars2, fashion_error_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('error_rates_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_comparison_chart(self):
        """生成性能比較圖表"""
        print("生成性能比較圖表...")
        
        # 實驗數據
        methods = ['原始數據', '正規化', '最佳超參數', '最終優化']
        mnist_scores = [86.70, 91.03, 91.54, 91.74]
        fashion_scores = [80.64, 84.13, 85.37, 82.31]
        
        x = np.arange(len(methods))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars1 = ax.bar(x - width/2, mnist_scores, width, label='MNIST', 
                      color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, fashion_scores, width, label='Fashion MNIST', 
                      color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('優化方法', fontsize=12, fontweight='bold')
        ax.set_ylabel('準確率 (%)', fontsize=12, fontweight='bold')
        ax.set_title('MNIST vs Fashion MNIST 性能比較', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_all_visualizations(self):
        """運行所有可視化"""
        self.load_datasets()
        
        # 生成混淆矩陣
        conf_mx_mnist, conf_mx_fashion = self.generate_confusion_matrices()
        
        # 分析混淆情況
        mnist_confused = self.find_confused_pairs(conf_mx_mnist, "MNIST")
        fashion_confused = self.find_confused_pairs(conf_mx_fashion, "Fashion MNIST")
        
        # 生成混淆樣本圖
        self.visualize_confused_digits(mnist_confused)
        self.visualize_confused_fashion(fashion_confused)
        
        # 生成錯誤率分析圖
        self.generate_error_analysis_chart(conf_mx_mnist, conf_mx_fashion)
        
        # 生成性能比較圖
        self.generate_performance_comparison_chart()
        
        print("\n所有可視化圖片已生成完成！")
        print("生成的圖片文件:")
        print("1. confusion_matrices.png - 混淆矩陣")
        print("2. confused_mnist_digits.png - MNIST混淆數字樣本")
        print("3. confused_fashion_items.png - Fashion MNIST混淆樣本")
        print("4. error_rates_comparison.png - 錯誤率比較")
        print("5. performance_comparison.png - 性能比較")

if __name__ == "__main__":
    generator = VisualizationGenerator()
    generator.run_all_visualizations()