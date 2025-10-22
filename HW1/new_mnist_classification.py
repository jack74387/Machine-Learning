"""
Homework-1: Multiclass Classification
Part 1: MNIST Dataset Classification using SGDClassifier

Author: Kiro AI Assistant
Date: 2025-10-13
"""

import numpy as np
import matplotlib.pyplot as plt                 #沒用到
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
# 引入 cross_val_score（計算 CV 分數）與 cross_val_predict（取得交叉驗證下的預測）
from sklearn.model_selection import cross_val_score, cross_val_predict
# 引入混淆矩陣與分類報告工具
from sklearn.metrics import confusion_matrix, classification_report
from scipy.ndimage import shift, rotate         # 引入 shift 函數，用於影像平移（資料擴增）
import warnings                                 # 引入 warnings 模組，用以抑制不需要的警告
warnings.filterwarnings('ignore')               # 忽略所有警告（避免執行時被大量警告淹沒）

class MNISTClassifier:
    def __init__(self):
        self.X_train = None         # 訓練集特徵（預設為 None，稍後填入 numpy 陣列）
        self.y_train = None         # 訓練集標籤
        self.X_test = None          # 測試集特徵
        self.y_test = None          # 測試集標籤
        self.sgd_clf = None         # 存放已訓練或待訓練的 SGDClassifier 實例
        
    def load_mnist_data(self):
        """載入MNIST數據集"""
        print("Loading MNIST dataset...")
        # 下載 MNIST（784 維平面化），as_frame=False 表示回傳 numpy 格式（非 pandas DataFrame） 
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        
        # mnist["data"]是 影像的像素資料 (features)，形狀是 (70000, 784)   784=28*28
        # 每一列是一張手寫數字圖（展平成一維），每個元素是一個像素的灰階值（0~255）
        # e.g.  mnist["data"][0]  # 第一張圖片的像素值

        # mnist["target"]是 每張圖片的真實標籤 (label)，形狀是 (70000,)
        # 每個值是對應的數字（0～9），預設型別是字串 (str)，所以要轉成整數
        # e.g. mnist["target"][0:10]
        # => ['5', '0', '4', '1', '9', '2', '1', '3', '1', '4']
        X, y = mnist["data"], mnist["target"].astype(np.uint8) #astype:轉換型態成np.uint8

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
        self.sgd_clf = SGDClassifier(random_state=42)   # random_state 固定隨機性，方便實驗可重複
        
        # 執行3折交叉驗證
        # cross_val_score 會把訓練集切成 3 折，返回每一折的 accuracy 分數
        print("Performing 3-fold cross-validation...")
        cv_scores = cross_val_score(self.sgd_clf, self.X_train, self.y_train, 
                                   cv=3, scoring="accuracy")
        
        print(f"Cross-validation scores: {cv_scores}")      # 顯示每一折的分數
        print(f"Mean accuracy: {cv_scores.mean():.4f}")     # 顯示平均準確率（四位小數）
        print(f"Standard deviation: {cv_scores.std():.4f}") # 顯示標準差
        
        # 訓練完整模型用於後續任務
        # 在整個訓練集上訓練一個最終模型，以便之後做預測或混淆矩陣分析
        print("Training final model on full training set...")
        self.sgd_clf.fit(self.X_train, self.y_train)

        
        return cv_scores.mean()     # 回傳交叉驗證的平均準確率

    #NEW ADD
    def random_shift_image(self, image):
        """隨機平移 -1~1 像素"""
        if np.random.randint(0, 2) == 0:
            dx = np.random.randint(-1, 2)
            dy = 0
        else:
            dy = np.random.randint(-1, 2)
            dx = 0
        image = image.reshape(28, 28)
        shifted = shift(image, [dy, dx], cval=0, mode='constant')
        return shifted.reshape(784)

    def random_rotate_image(self, image):
        """隨機旋轉 ±(3、6) 度"""
        angle = np.random.choice([2, 4]) * np.random.choice([-1, 1])
        image = image.reshape(28, 28)
        rotated = rotate(image, angle, reshape=False, cval=0, mode='constant')
        return rotated.reshape(784)



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
        image = image.reshape(28, 28)               # 將一維向量重塑為 28x28 的二維矩陣
        # 使用 scipy.ndimage.shift 做位移，mode='constant' 並以 cval=0 填補空白（黑色）
        shifted = shift(image, [dy, dx], cval=0, mode='constant')
        return shifted.reshape(784)                 # 把結果再攤平成一維向量返回
    
    def augment_dataset(self, X, y, augment_size=10000, method="shift"):
        """產生指定方法與數量的增強資料"""
        print(f"Generating {augment_size} augmented samples with method '{method}'...")
        
        # 隨機選擇要增強的樣本（可重複抽樣）
        indices = np.random.choice(len(X), augment_size, replace=True)
        
        augmented_X, augmented_y = [], []
        
        #directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 左、右、上、下
        
        for i, idx in enumerate(indices):
            img = X[idx]
            if method == "shift":
                aug_img = self.random_shift_image(img)
            elif method == "rotate":
                aug_img = self.random_rotate_image(img)
            elif method == "mixed":
                if np.random.rand() < 0.5:
                    aug_img = self.random_shift_image(img)
                else:
                    aug_img = self.random_rotate_image(img)
            else:
                raise ValueError(f"Unknown augmentation method: {method}")
            # 生成擴增樣本（對選到的樣本做平移）
            #augmented_image = self.shift_image(X[idx], dx, dy)
            augmented_X.append(aug_img)     # 加入擴增後的影像
            augmented_y.append(y[idx])              # 標籤與原圖相同

            if (i + 1) % 4000 == 0:
                print(f"Progress: {i+1}/{augment_size}")
        
        # 回傳 numpy 陣列形式的增強資料與標籤，以利後續訓練使用
        return np.array(augmented_X), np.array(augmented_y)
    
    def task_1_2_data_augmentation(self):
        """
        Task 1.2: 強化版資料增強實驗
        比較三種增強方式 × 三種樣本數
        """
        print("\n" + "="*50)
        print("Task 1.2: Enhanced Data Augmentation Experiments")
        print("="*50)
        
        methods = ["shift", "rotate", "mixed"]
        augment_sizes = [8000, 10000, 12000]

        results = {}

        for method in methods:
            results[method] = {}
            for size in augment_sizes:
                print(f"\n[Experiment] Method = {method}, Size = {size}")
                aug_X, aug_y = self.augment_dataset(self.X_train, self.y_train, augment_size=size, method=method)
                
                X_combined = np.vstack([self.X_train, aug_X])
                y_combined = np.hstack([self.y_train, aug_y])
                
                sgd = SGDClassifier(random_state=42)
                cv_scores = cross_val_score(sgd, X_combined, y_combined, cv=3, scoring="accuracy")
                
                mean_acc = cv_scores.mean()
                print(f"CV scores: {cv_scores}")
                print(f"Mean accuracy: {mean_acc:.4f}")
                
                results[method][size] = mean_acc
        
        print("\n=== Data Augmentation Summary ===")
        for method in methods:
            for size in augment_sizes:
                print(f"Method={method:6} | Size={size:6} | Accuracy={results[method][size]:.4f}")
        
        # 找出最佳設定
        best_method, best_size = max(
            ((m, s) for m in methods for s in augment_sizes),
            key=lambda x: results[x[0]][x[1]]
        )
        
        print(f"\nBest augmentation: method={best_method}, size={best_size}, "
            f"accuracy={results[best_method][best_size]:.4f}")

        return results         
    
    def task_1_3_confusion_matrix_analysis(self):
        """
        Task 1.3: 使用混淆矩陣分析並改善效能
        """
        print("\n" + "="*50)
        print("Task 1.3: Confusion Matrix Analysis and Performance Improvement")
        print("="*50)
        
         # 使用基礎模型進行預測（cross_val_predict 會在交叉驗證的每一折上產生對應的預測，避免資料外洩）
        y_train_pred = cross_val_predict(self.sgd_clf, self.X_train, self.y_train, cv=3)
        
        # 生成混淆矩陣（用於分析哪些類別被混淆）
        conf_mx = confusion_matrix(self.y_train, y_train_pred)
        print("Confusion Matrix:")
        print(conf_mx)      # 列印出混淆矩陣（行：真實類別，列：預測類別）
        
        # 計算每個類別的錯誤率，逐類別分析
        print("\nError analysis by class:")
        for i in range(10):                                         # 對 0~9 共 10 類別逐一分析
            total_samples = np.sum(conf_mx[i, :])                   # 該真實類別的所有樣本數（橫向總和）
            correct_predictions = conf_mx[i, i]                     # 該類別預測正確的數量（對角元素）
            # 錯誤率 = 1 - 正確率（或直接計算錯誤數 / 總數）
            error_rate = (total_samples - correct_predictions) / total_samples
            print(f"Class {i}: Error rate = {error_rate:.4f}")
        
        # 嘗試正規化改善（將像素值縮放到 0-1）
        print("\nTrying normalization...")
        X_train_normalized = self.X_train / 255.0       # 除以 255，把像素範圍從 [0,255] 轉為 [0,1]
        
        sgd_normalized = SGDClassifier(random_state=42)        # 新的 SGD 實例，用於在正規化後評估 
        cv_scores_normalized = cross_val_score(sgd_normalized, X_train_normalized, 
                                             self.y_train, cv=3, scoring="accuracy")
        # 在正規化後的資料上做交叉驗證，觀察效果
        
        print(f"Normalized CV scores: {cv_scores_normalized}")
        print(f"Normalized mean accuracy: {cv_scores_normalized.mean():.4f}")
        
        # 嘗試超參數調整（例如 alpha、max_iter）
        print("\nTrying hyperparameter tuning...")
        sgd_tuned = SGDClassifier(
            alpha=0.001,    # 較小的 L2/L1 正規化強度（視預設 penalty 而定）
            max_iter=1000,  # 增加最大疊代次數，讓 SGD 有更多收斂機會
            random_state=42
        )
        
        cv_scores_tuned = cross_val_score(sgd_tuned, X_train_normalized, 
                                        self.y_train, cv=3, scoring="accuracy")
        # 在相同正規化資料上測試調整後的參數

        print(f"Tuned CV scores: {cv_scores_tuned}")
        print(f"Tuned mean accuracy: {cv_scores_tuned.mean():.4f}")
        
        # 保存最佳模型（這裡把 sgd_tuned 當作最佳模型並以全部正規化訓練資料做 fit）
        sgd_tuned.fit(X_train_normalized, self.y_train)
        self.sgd_clf_best = sgd_tuned           # 儲存最佳模型供測試階段使用
        self.X_train_normalized = X_train_normalized        # 儲存正規化後的訓練資料
        
        return cv_scores_tuned.mean()           # 回傳調參後的平均交叉驗證準確率
    
    def evaluate_on_test_set(self):
        """在測試集上評估最佳模型"""
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        
        # 正規化測試集（與訓練集相同的前處理）
        X_test_normalized = self.X_test / 255.0
        
        ## 使用最佳模型進行預測（確保 self.sgd_clf_best 已存在）
        y_test_pred = self.sgd_clf_best.predict(X_test_normalized)
        
        # 計算準確率（測試集）
        test_accuracy = np.mean(y_test_pred == self.y_test)
        print(f"Test set accuracy: {test_accuracy:.4f}")
        
         # 生成並列印詳細的分類報告（precision, recall, f1-score, support）
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred))
        
        return test_accuracy    # 回傳測試集準確率

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
    print(f"\nTask 1.2 completed with accuracy: {accuracy_augmented}")
    
    # 執行Task 1.3
    accuracy_optimized = classifier.task_1_3_confusion_matrix_analysis()
    print(f"\nTask 1.3 completed with accuracy: {accuracy_optimized:.4f}")
    
    # 最終測試集評估
    test_accuracy = classifier.evaluate_on_test_set()
    
    # 總結結果
    print("\n" + "="*50)
    print("PART 1 SUMMARY")
    print("="*50)
    print(f"Basic SGD accuracy: {accuracy_basic:.4f}")
    print(f"With data augmentation: {accuracy_augmented:.4f}")
    print(f"With optimization: {accuracy_optimized:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()