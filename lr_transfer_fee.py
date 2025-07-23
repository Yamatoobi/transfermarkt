import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
# Merge club info to get domestic_competition_id
df = df.merge(clubs[["club_id", "domestic_competition_id"]], on="club_id", how="left")

# Big5 league IDs
big5_ids = ["GB1", "ES1", "IT1", "L1", "FR1"]
df["is_big5_league"] = df["domestic_competition_id"].isin(big5_ids).astype(int)

# 2. Prepare features
performance_features = ["performance_composite_score", "position_composite_score"]
non_performance_features = [
    "age",
    "height_in_cm",
    "is_big5_league",
]

# One-hot encode 'foot'
df = pd.get_dummies(df, columns=["foot"], drop_first=True)

# Group country_of_citizenship into Top10/Other and one-hot encode
top10_countries = ['France', 'Germany', 'Spain', 'Italy', 'Brazil', 'Argentina', 'England', 'Portugal', 'Netherlands', 'Croatia']
df["country_group"] = df["country_of_citizenship"].apply(lambda x: "Top10" if x in top10_countries else "Other")
df = pd.get_dummies(df, columns=["country_group"], drop_first=True)

# 3. Select features for regression
features = performance_features + non_performance_features
features += [col for col in df.columns if col.startswith("foot_") or col.startswith("country_group_")]

# 4. Drop rows with missing values in features or target, and transfer_fee <= 0
df_reg = df.dropna(subset=features + ["transfer_fee"])
df_reg = df_reg[df_reg["transfer_fee"] > 0]

# 5. Prepare X and y
X = df_reg[features]
y = np.log(df_reg["transfer_fee"])

# 6. Fit regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

# 7. Output results
print("R² score:", r2)
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")

# Optionally, save results
results_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
})
results_df.to_csv("transfer_fee_regression_coefficients.csv", index=False)
print("Saved regression coefficients to 'transfer_fee_regression_coefficients.csv'.")