import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# ========== Load data ==========
df = pd.read_csv("panel_df_cleaned.csv")

# ========== Define position groups ==========
position_groups = {
    "Attackers": ["Centre-Forward", "Right Winger", "Left Winger", "Second Striker"],
    "Midfielders": ["Attacking Midfield", "Central Midfield", "Defensive Midfield"],
    "Defenders": ["Centre-Back", "Right-Back", "Left-Back", "Full-Back"],
    "Goalkeepers": ["Keeper", "Goalkeeper"]
}

# ========== Define performance features ==========
from main_pca import pca_features

# ========== Create output folder ==========
os.makedirs("pca_outputs_by_position", exist_ok=True)

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
    
    print(f"{group_name}: PCA complete. Explained variance ratios:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {v:.2f}")



# Add position-specific PC1 to panel_with_pca
combined_df = pd.read_csv("panel_with_pca.csv")

for group_name in position_groups.keys():
    pcs_path = f"pca_outputs_by_position/{group_name}_with_pcs.csv"
    if os.path.exists(pcs_path):
        pcs_df = pd.read_csv(pcs_path)
        # Only keep player_id, year, and available PC columns
        pcs_columns = ["player_id", "year"] + [f"PC{i+1}" for i in range(5) if f"PC{i+1}" in pcs_df.columns]
        pcs_df = pcs_df[pcs_columns]
        # Rename PC columns to include group name
        pcs_df = pcs_df.rename(columns={f"PC{i+1}": f"{group_name}_PC{i+1}" for i in range(5) if f"PC{i+1}" in pcs_df.columns})
        # Merge into combined_df
        combined_df = combined_df.merge(pcs_df, on=["player_id", "year"], how="left")
combined_df.to_csv("panel_with_all_pca.csv", index=False)
print("Saved full panel with position-specific PC1~5 to 'panel_with_all_pca.csv'.")