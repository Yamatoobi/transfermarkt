import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# 1. Load cleaned panel and composite scores
df = pd.read_csv("panel_df_cleaned.csv")
pca_df = pd.read_csv("panel_with_all_pca.csv")
clubs = pd.read_csv("clubs.csv")

# Merge composite scores into cleaned panel (on player_id and year)
composite_cols = [
    "performance_composite_score",
    "Attackers_composite_score",
    "Midfielders_composite_score",
    "Defenders_composite_score",
    "Goalkeepers_composite_score"
]
merge_cols = ["player_id", "year"]

df = df.merge(
    pca_df[merge_cols + composite_cols],
    on=merge_cols,
    how="left"
)

# Create a single position_composite_score column (the non-NaN one for each row)
df["position_composite_score"] = df[
    ["Attackers_composite_score", "Midfielders_composite_score", "Defenders_composite_score", "Goalkeepers_composite_score"]
].bfill(axis=1).iloc[:, 0]

# ========== Add Big5 league flag ==========
df = df.merge(clubs[["club_id", "domestic_competition_id"]], on="club_id", how="left")
big5_ids = ["GB1", "ES1", "IT1", "L1", "FR1"]
df["is_big5_league"] = df["domestic_competition_id"].isin(big5_ids).astype(int)

# One-hot encode 'foot'
df = pd.get_dummies(df, columns=["foot"], drop_first=True)

# Group country_of_citizenship into Top10/Other and one-hot encode
top10_countries = ['France', 'Germany', 'Spain', 'Italy', 'Brazil', 'Argentina', 'England', 'Portugal', 'Netherlands', 'Croatia']
df["country_group"] = df["country_of_citizenship"].apply(lambda x: "Top10" if x in top10_countries else "Other")
df = pd.get_dummies(df, columns=["country_group"], drop_first=True)

# ========== 回帰分析（yearなし） ==========
performance_features = ["performance_composite_score", "position_composite_score"]
non_performance_features = [
    "age",
    "height_in_cm",
    "is_big5_league",
]
features = performance_features + non_performance_features
features += [col for col in df.columns if col.startswith("foot_") or col.startswith("country_group_")]

df_reg = df.dropna(subset=features + ["market_value_in_eur"])
X = df_reg[features]
y = np.log(df_reg["market_value_in_eur"].replace(0, np.nan))

# 標準化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_std = scaler_X.fit_transform(X)
y_std = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

model = LinearRegression()
model.fit(X_std, y_std)
y_pred = model.predict(X_std)
r2 = r2_score(y_std, y_pred)

print("Standardized regression (market value, no year)")
print("R² score:", r2)
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")

results_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})
results_df["R2"] = r2
results_df.to_csv("market_value_regression_coefficients.csv", index=False)
print("Saved regression coefficients to 'market_value_regression_coefficients.csv'.")

# ========== 回帰分析（yearあり） ==========
features_with_year = features + ["year"]
df_reg2 = df.dropna(subset=features_with_year + ["market_value_in_eur"])
X2 = df_reg2[features_with_year]
y2 = np.log(df_reg2["market_value_in_eur"].replace(0, np.nan))

scaler_X2 = StandardScaler()
scaler_y2 = StandardScaler()
X2_std = scaler_X2.fit_transform(X2)
y2_std = scaler_y2.fit_transform(y2.values.reshape(-1, 1)).ravel()

model2 = LinearRegression()
model2.fit(X2_std, y2_std)
y2_pred = model2.predict(X2_std)
r2_2 = r2_score(y2_std, y2_pred)

print("Standardized regression (market value, with year)")
print("R² score:", r2_2)
for name, coef in zip(X2.columns, model2.coef_):
    print(f"  {name}: {coef:.4f}")

results_df2 = pd.DataFrame({
    "feature": X2.columns,
    "coefficient": model2.coef_
})
results_df2["R2"] = r2_2
results_df2.to_csv("market_value_regression_coefficients_with_year.csv", index=False)
print("Saved regression coefficients to 'market_value_regression_coefficients_with_year.csv'.")

# ==================== transfer_fee ====================
# filepath: /Users/yamatoobinata/Downloads/ML/tansfrmarket/lr_transfer_fee.py
# ここから下をlr_transfer_fee.pyとして保存してもOK

# 1. transfer_fee（yearなし）
df_reg = df.dropna(subset=features + ["transfer_fee"])
df_reg = df_reg[df_reg["transfer_fee"] > 0]
X = df_reg[features]
y = np.log(df_reg["transfer_fee"])

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_std = scaler_X.fit_transform(X)
y_std = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

model = LinearRegression()
model.fit(X_std, y_std)
y_pred = model.predict(X_std)
r2 = r2_score(y_std, y_pred)

print("Standardized regression (transfer fee, no year)")
print("R² score:", r2)
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")

results_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})
results_df["R2"] = r2
results_df.to_csv("transfer_fee_regression_coefficients.csv", index=False)
print("Saved regression coefficients to 'transfer_fee_regression_coefficients.csv'.")

# 2. transfer_fee（yearあり）
df_reg2 = df.dropna(subset=features_with_year + ["transfer_fee"])
df_reg2 = df_reg2[df_reg2["transfer_fee"] > 0]
X2 = df_reg2[features_with_year]
y2 = np.log(df_reg2["transfer_fee"])

scaler_X2 = StandardScaler()
scaler_y2 = StandardScaler()
X2_std = scaler_X2.fit_transform(X2)
y2_std = scaler_y2.fit_transform(y2.values.reshape(-1, 1)).ravel()

model2 = LinearRegression()
model2.fit(X2_std, y2_std)
y2_pred = model2.predict(X2_std)
r2_2 = r2_score(y2_std, y2_pred)

print("Standardized regression (transfer fee, with year)")
print("R² score:", r2_2)
for name, coef in zip(X2.columns, model2.coef_):
    print(f"  {name}: {coef:.4f}")

results_df2 = pd.DataFrame({
    "feature": X2.columns,
    "coefficient": model2.coef_
})
results_df2["R2"] = r2_2
results_df2.to_csv("transfer_fee_regression_coefficients_with_year.csv", index=False)
print("Saved regression coefficients to 'transfer_fee_regression_coefficients_with_year.csv'.")