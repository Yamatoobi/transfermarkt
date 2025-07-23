# we will use panel_df.csv and clean data

import pandas as pd

panel_df = pd.read_csv("panel_df.csv")

# Drop rows with critical missing values (e.g., minutes_played is essential)
before = len(panel_df)
panel_df_cleaned = panel_df.dropna(subset=["minutes_played", "goals", "assists"])
after = len(panel_df_cleaned)
print(f"Dropped {before - after} rows due to missing critical values.")

# 2. Fill or flag less critical fields
# Fill red_cards and yellow_cards with 0 if missing (assumption: no card)
panel_df_cleaned["yellow_cards"] = panel_df_cleaned["yellow_cards"].fillna(0).astype(int)
panel_df_cleaned["red_cards"] = panel_df_cleaned["red_cards"].fillna(0).astype(int)

# Fill market_value and transfer_fee with a placeholder or keep NaN depending on purpose
# For analysis where presence of value is necessary:
# panel_df_cleaned = panel_df_cleaned.dropna(subset=["market_value_in_eur"])

# Alternate option: fill with 0 (if "no value" is valid)
# panel_df_cleaned["market_value_in_eur"] = panel_df_cleaned["market_value_in_eur"].fillna(0)
# panel_df_cleaned["transfer_fee"] = panel_df_cleaned["transfer_fee"].fillna(0)

# Fill categorical info if desired
panel_df_cleaned["position"] = panel_df_cleaned["position"].fillna("Unknown")

# Fill missing values with 0 for new columns
for col in ["defensive_contributions", "starts", "subs", "saves", "penalty_saves"]:
    if col in panel_df_cleaned.columns:
        panel_df_cleaned[col] = panel_df_cleaned[col].fillna(0).astype(int)

# Add club_id, club name, and total_market_value columns if not already present
# (Assumes these were added in create_panel_df.py)
for col in ["club_id", "name", "total_market_value"]:
    if col not in panel_df_cleaned.columns and col in panel_df.columns:
        panel_df_cleaned[col] = panel_df[col]

# Output result
panel_df_cleaned.to_csv("panel_df_cleaned.csv", index=False)
print("Cleaned panel_df saved to 'panel_df_cleaned.csv'.")