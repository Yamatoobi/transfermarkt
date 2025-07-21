import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the cleaned panel
panel_df = pd.read_csv("panel_df_cleaned.csv")

# Select performance-related features for PCA
pca_features = [
    "minutes_played", "goals", "assists",
    "yellow_cards", "red_cards", "appearances",
    "goals_per_90", "assists_per_90",
    "starts", "subs"
]

# Replace missing values with 0 (interpreted as 'no performance')
X = panel_df[pca_features].fillna(0)

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run PCA
pca = PCA(n_components=5)
principal_components = pca.fit_transform(X_scaled)

# Add PC1, PC2, PC3 to dataframe
panel_df["performance_score"] = principal_components[:, 0]
panel_df["performance_score_2"] = principal_components[:, 1]
panel_df["performance_score_3"] = principal_components[:, 2]
panel_df["performance_score_4"] = principal_components[:, 3]
panel_df["performance_score_5"] = principal_components[:, 4]

# Print explained variance for interpretability check
print("Explained variance ratios:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

# Optional: print component loadings for interpretation
loadings = pd.DataFrame(
    pca.components_.T,
    index=pca_features,
    columns=["PC1", "PC2", "PC3", "PC4", "PC5"]
)
print("\nPCA Component Loadings:")
print(loadings)

# Save the updated dataframe with scores
panel_df.to_csv("panel_with_pca.csv", index=False)
print("Saved PCA results to 'panel_with_pca.csv'.")