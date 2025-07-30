import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# データ読み込み
df = pd.read_csv("panel_with_all_pca.csv")

# 各ポジション別に多重共線性をチェック
position_scores = [
    "Attackers_composite_score", 
    "Midfielders_composite_score", 
    "Defenders_composite_score", 
    "Goalkeepers_composite_score"
]

for pos_score in position_scores:
    # 該当ポジションのデータのみ（NaNでないもの）
    X = df[["performance_composite_score", pos_score]].dropna()
    
    if len(X) > 0:
        # 標準化
        X_std = StandardScaler().fit_transform(X)
        
        # VIF計算
        vif_df = pd.DataFrame()
        vif_df["variable"] = ["performance_composite_score", pos_score]
        vif_df["VIF"] = [variance_inflation_factor(X_std, i) for i in range(X_std.shape[1])]
        
        print(f"\n{pos_score}:")
        print(vif_df)
    else:
        print(f"\n{pos_score}: No data available")