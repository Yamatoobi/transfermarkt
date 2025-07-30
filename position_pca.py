import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import numpy as np

# ========== Load data ==========
df = pd.read_csv("panel_df_cleaned.csv")

# ========== Define position groups ==========
position_groups = {
    "Attackers": ["Centre-Forward", "Right Winger", "Left Winger", "Second Striker"],
    "Midfielders": ["Attacking Midfield", "Central Midfield", "Defensive Midfield", "Right Midfield", "Left Midfield"],
    "Defenders": ["Centre-Back", "Right-Back", "Left-Back"],
    "Goalkeepers": ["Keeper", "Goalkeeper"]
}

# ========== Define features ==========
from main_pca import pca_features 

# ========== Process each group ==========
for group_name, positions in position_groups.items():
    group_df = df[df["position"].isin(positions)].copy()
    
    # Skip if too few players
    if group_df.shape[0] < 50:
        print(f"Skipping {group_name}: too few samples")
        continue

    X = group_df[pca_features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=min(len(pca_features), 5))
    components = pca.fit_transform(X_scaled)
    loadings = pd.DataFrame(pca.components_.T, index=pca_features, columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    
    # Calculate weighted composite score
    weights = pca.explained_variance_ratio_
    composite_score = np.zeros(len(X))
    for i in range(pca.n_components_):
        composite_score += weights[i] * components[:, i]
    
    # Add composite score to group_df
    group_df[f"{group_name}_composite_score"] = composite_score
    
    # Save loadings
    loadings.to_csv(f"pca_outputs_by_position/{group_name}_loadings.csv")
    
    # Save explained variance
    with open(f"pca_outputs_by_position/{group_name}_variance.txt", "w") as f:
        for i, v in enumerate(pca.explained_variance_ratio_):
            f.write(f"PC{i+1}: {v:.4f}\n")
    
    # Add component scores to df for inspection
    for i in range(pca.n_components_):
        group_df[f"PC{i+1}"] = components[:, i]
    
    # Save the group-level PCA result
    group_df.to_csv(f"pca_outputs_by_position/{group_name}_with_pcs.csv", index=False)
    
    # Print explained variance ratios for PC1~PC5
    print(f"\n{group_name} PCA Explained Variance Ratios:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v:.4f}")

    # Print component loadings for PC1~PC5
    print(f"\n{group_name} PCA Component Loadings:")
    print(loadings.iloc[:, :5].round(3))  # Show up to PC5, rounded for readability

# Add position-specific PCs and composite scores to panel_with_pca
combined_df = pd.read_csv("panel_with_pca.csv")

for group_name in position_groups.keys():
    pcs_path = f"pca_outputs_by_position/{group_name}_with_pcs.csv"
    if os.path.exists(pcs_path):
        pcs_df = pd.read_csv(pcs_path)
        # Keep player_id, year, PCs and composite score
        keep_cols = ["player_id", "year"] + [f"PC{i+1}" for i in range(5) if f"PC{i+1}" in pcs_df.columns]
        if f"{group_name}_composite_score" in pcs_df.columns:
            keep_cols.append(f"{group_name}_composite_score")
        pcs_df = pcs_df[keep_cols]
        # Rename PC columns to include group name
        pcs_df = pcs_df.rename(columns={f"PC{i+1}": f"{group_name}_PC{i+1}" for i in range(5) if f"PC{i+1}" in pcs_df.columns})
        # Merge into combined_df
        combined_df = combined_df.merge(pcs_df, on=["player_id", "year"], how="left")

combined_df.to_csv("panel_with_all_pca.csv", index=False)
print("Saved full panel with position-specific PCs and composite scores to 'panel_with_all_pca.csv'.")