import pandas as pd
import numpy as np

# データの読み込み
panel_pca = pd.read_csv("panel_with_all_pca.csv")

# 利用可能な列を確認
print("Available columns:")
print(panel_pca.columns.tolist())

# ポジション分布を確認
print("\nPosition distribution:")
print(panel_pca['position'].value_counts())

# PCスコアの列を確認
pc_cols = [col for col in panel_pca.columns if 'PC' in col or 'performance_score' in col]
print("\nPC score columns:")
print(pc_cols)