"""
TMDB 電影票房預測 - 探索性資料分析與視覺化
完成 task.md Phase 2 的視覺化任務
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體（如果需要）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 設定視覺化風格
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 60)
print("TMDB 電影票房預測 - 探索性資料分析")
print("=" * 60)

# 載入資料
print("\n[1/7] 載入資料...")
train_df = pd.read_csv('train.csv')
print(f"✓ 訓練集大小: {train_df.shape}")

# 1. 目標變數分析
print("\n[2/7] 分析目標變數 revenue...")
print(f"Revenue 統計:")
print(f"  平均值: ${train_df['revenue'].mean():,.0f}")
print(f"  中位數: ${train_df['revenue'].median():,.0f}")
print(f"  標準差: ${train_df['revenue'].std():,.0f}")
print(f"  最小值: ${train_df['revenue'].min():,.0f}")
print(f"  最大值: ${train_df['revenue'].max():,.0f}")

# 視覺化 1: Revenue 分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_df['revenue'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Revenue Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Revenue ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(train_df['revenue'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${train_df["revenue"].mean():,.0f}')
axes[0].axvline(train_df['revenue'].median(), color='green', linestyle='--', 
                label=f'Median: ${train_df["revenue"].median():,.0f}')
axes[0].legend()

axes[1].hist(np.log1p(train_df['revenue']), bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_title('Log(Revenue) Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Log(Revenue)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.savefig('eda_1_revenue_distribution.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已儲存: eda_1_revenue_distribution.png")
plt.close()

# 2. 數值特徵分析
print("\n[3/7] 分析數值特徵...")
numerical_features = ['budget', 'popularity', 'runtime']

# 視覺化 2: 數值特徵與 revenue 的關係
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for idx, feature in enumerate(numerical_features):
    # 散點圖
    axes[0, idx].scatter(train_df[feature], train_df['revenue'], alpha=0.3, s=10)
    axes[0, idx].set_xlabel(feature.capitalize(), fontsize=12)
    axes[0, idx].set_ylabel('Revenue ($)', fontsize=12)
    axes[0, idx].set_title(f'{feature.capitalize()} vs Revenue', fontsize=13, fontweight='bold')
    
    # 計算相關係數
    corr = train_df[[feature, 'revenue']].corr().iloc[0, 1]
    axes[0, idx].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                      transform=axes[0, idx].transAxes, 
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                      verticalalignment='top')
    
    # 箱型圖
    axes[1, idx].boxplot(train_df[feature].dropna(), vert=True)
    axes[1, idx].set_ylabel(feature.capitalize(), fontsize=12)
    axes[1, idx].set_title(f'{feature.capitalize()} Distribution', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('eda_2_numerical_features.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已儲存: eda_2_numerical_features.png")
plt.close()

# 3. 相關性矩陣
print("\n[4/7] 分析特徵相關性...")
correlation_features = ['budget', 'popularity', 'runtime', 'revenue']
correlation_matrix = train_df[correlation_features].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_3_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已儲存: eda_3_correlation_matrix.png")
plt.close()

# 4. 時間趨勢分析
print("\n[5/7] 分析時間趨勢...")
train_df['release_date'] = pd.to_datetime(train_df['release_date'], errors='coerce')
train_df['release_year'] = train_df['release_date'].dt.year
train_df['release_month'] = train_df['release_date'].dt.month

# 視覺化 4: 年份與月份趨勢
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 年份趨勢
yearly_revenue = train_df.groupby('release_year')['revenue'].agg(['mean', 'count'])
yearly_revenue = yearly_revenue[yearly_revenue['count'] >= 5]  # 至少5部電影

axes[0].plot(yearly_revenue.index, yearly_revenue['mean'], marker='o', linewidth=2)
axes[0].set_xlabel('Release Year', fontsize=12)
axes[0].set_ylabel('Average Revenue ($)', fontsize=12)
axes[0].set_title('Average Revenue by Year', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# 月份趨勢
monthly_revenue = train_df.groupby('release_month')['revenue'].mean()
axes[1].bar(monthly_revenue.index, monthly_revenue.values, color='skyblue', edgecolor='black')
axes[1].set_xlabel('Release Month', fontsize=12)
axes[1].set_ylabel('Average Revenue ($)', fontsize=12)
axes[1].set_title('Average Revenue by Month', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, 13))
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('eda_4_time_trends.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已儲存: eda_4_time_trends.png")
plt.close()

# 5. 類別特徵分析 - Genres
print("\n[6/7] 分析類別特徵...")

def parse_json_first_name(x):
    try:
        if pd.isna(x) or x == '':
            return 'Unknown'
        data = json.loads(x.replace("'", '"'))
        if isinstance(data, list) and len(data) > 0:
            return data[0].get('name', 'Unknown')
        return 'Unknown'
    except:
        return 'Unknown'

train_df['top_genre'] = train_df['genres'].apply(parse_json_first_name)

# 視覺化 5: Top 10 類型的平均票房
genre_revenue = train_df.groupby('top_genre')['revenue'].agg(['mean', 'count'])
genre_revenue = genre_revenue[genre_revenue['count'] >= 10]  # 至少10部電影
genre_revenue = genre_revenue.sort_values('mean', ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(range(len(genre_revenue)), genre_revenue['mean'], color='coral', edgecolor='black')
ax.set_yticks(range(len(genre_revenue)))
ax.set_yticklabels(genre_revenue.index)
ax.set_xlabel('Average Revenue ($)', fontsize=12)
ax.set_title('Top 10 Genres by Average Revenue', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# 添加數值標籤
for i, (idx, row) in enumerate(genre_revenue.iterrows()):
    ax.text(row['mean'], i, f" ${row['mean']:,.0f} ({int(row['count'])} films)", 
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig('eda_5_genre_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已儲存: eda_5_genre_analysis.png")
plt.close()

# 6. Budget vs Revenue 深入分析
print("\n[7/7] 深入分析 Budget vs Revenue...")

# 視覺化 6: Budget vs Revenue 分組分析
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 散點圖 + 趨勢線
axes[0, 0].scatter(train_df['budget'], train_df['revenue'], alpha=0.3, s=20)
z = np.polyfit(train_df['budget'].fillna(0), train_df['revenue'], 1)
p = np.poly1d(z)
axes[0, 0].plot(train_df['budget'].sort_values(), 
                p(train_df['budget'].sort_values()), 
                "r--", linewidth=2, label='Trend Line')
axes[0, 0].set_xlabel('Budget ($)', fontsize=12)
axes[0, 0].set_ylabel('Revenue ($)', fontsize=12)
axes[0, 0].set_title('Budget vs Revenue (with Trend)', fontsize=13, fontweight='bold')
axes[0, 0].legend()

# 2. Log-Log 散點圖
budget_nonzero = train_df[train_df['budget'] > 0]
axes[0, 1].scatter(np.log1p(budget_nonzero['budget']), 
                   np.log1p(budget_nonzero['revenue']), 
                   alpha=0.3, s=20, color='green')
axes[0, 1].set_xlabel('Log(Budget)', fontsize=12)
axes[0, 1].set_ylabel('Log(Revenue)', fontsize=12)
axes[0, 1].set_title('Log(Budget) vs Log(Revenue)', fontsize=13, fontweight='bold')

# 3. Budget 分組的平均 Revenue
budget_bins = [0, 1e6, 1e7, 5e7, 1e8, 2e8, 1e9]
budget_labels = ['<1M', '1M-10M', '10M-50M', '50M-100M', '100M-200M', '>200M']
train_df['budget_group'] = pd.cut(train_df['budget'], bins=budget_bins, labels=budget_labels)
budget_group_revenue = train_df.groupby('budget_group', observed=True)['revenue'].mean()

axes[1, 0].bar(range(len(budget_group_revenue)), budget_group_revenue.values, 
               color='purple', edgecolor='black', alpha=0.7)
axes[1, 0].set_xticks(range(len(budget_group_revenue)))
axes[1, 0].set_xticklabels(budget_group_revenue.index, rotation=45)
axes[1, 0].set_ylabel('Average Revenue ($)', fontsize=12)
axes[1, 0].set_title('Average Revenue by Budget Group', fontsize=13, fontweight='bold')

# 4. ROI 分析 (Return on Investment)
train_df['roi'] = (train_df['revenue'] - train_df['budget']) / (train_df['budget'] + 1)
roi_data = train_df[train_df['budget'] > 0]['roi']
axes[1, 1].hist(roi_data[roi_data < 20], bins=50, color='gold', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('ROI (Return on Investment)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('ROI Distribution', fontsize=13, fontweight='bold')
axes[1, 1].axvline(roi_data.median(), color='red', linestyle='--', 
                   label=f'Median ROI: {roi_data.median():.2f}')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('eda_6_budget_revenue_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 圖表已儲存: eda_6_budget_revenue_analysis.png")
plt.close()

# 生成 EDA 摘要報告
print("\n" + "=" * 60)
print("EDA 摘要報告")
print("=" * 60)

print("\n【關鍵發現】")
print(f"1. Revenue 分布呈現右偏態，中位數 (${train_df['revenue'].median():,.0f}) 遠小於平均值 (${train_df['revenue'].mean():,.0f})")
print(f"2. Budget 與 Revenue 的相關係數: {train_df[['budget', 'revenue']].corr().iloc[0, 1]:.3f} (強正相關)")
print(f"3. Popularity 與 Revenue 的相關係數: {train_df[['popularity', 'revenue']].corr().iloc[0, 1]:.3f}")
print(f"4. Runtime 與 Revenue 的相關係數: {train_df[['runtime', 'revenue']].corr().iloc[0, 1]:.3f}")

# 最佳上映月份
best_month = monthly_revenue.idxmax()
print(f"5. 最佳上映月份: {int(best_month)} 月 (平均票房: ${monthly_revenue[best_month]:,.0f})")

# 最佳類型
best_genre = genre_revenue['mean'].idxmax()
print(f"6. 最高票房類型: {best_genre} (平均票房: ${genre_revenue.loc[best_genre, 'mean']:,.0f})")

# ROI 統計
print(f"7. 平均 ROI: {roi_data.mean():.2f}x")
print(f"8. 中位數 ROI: {roi_data.median():.2f}x")

print("\n【建議】")
print("1. 考慮對 revenue 進行 log 轉換以處理偏態分布")
print("2. Budget 是最重要的預測特徵，應重點關注")
print("3. 可以創建 budget_group 作為類別特徵")
print("4. 時間特徵（月份、季度）可能有助於預測")
print("5. 類型特徵對票房有顯著影響")

print("\n" + "=" * 60)
print("EDA 完成！所有圖表已儲存。")
print("=" * 60)
print("\n生成的圖表:")
print("  - eda_1_revenue_distribution.png")
print("  - eda_2_numerical_features.png")
print("  - eda_3_correlation_matrix.png")
print("  - eda_4_time_trends.png")
print("  - eda_5_genre_analysis.png")
print("  - eda_6_budget_revenue_analysis.png")
