# This script restructures and merges football player data from multiple CSV files
import pandas as pd


# Restructuring data from Appearances.csv
df_app = pd.read_csv("Appearances.csv")
df_app["date"] = pd.to_datetime(df_app["date"])
df_app["year"] = df_app["date"].dt.year  # extract "year" from "date" as datetime object

# Sort player appearance data by year and player_id
player_yearly = df_app.groupby(["player_id", "year"]).agg({
    "player_name": "first",
    "minutes_played": "sum",
    "goals": "sum",
    "assists": "sum",
    "yellow_cards": "sum",
    "red_cards": "sum",
    "game_id": "count"
}).rename(columns={"game_id": "appearances"}).reset_index()

player_yearly["goals_per_90"] = player_yearly["goals"] / player_yearly["minutes_played"].replace(0, pd.NA) * 90 # normalizing goalscoring efficiency
player_yearly["assists_per_90"] = player_yearly["assists"] / player_yearly["minutes_played"].replace(0,pd.NA) * 90 # normalizing assist efficiency


# Restructuring data from player_valuations.csv
df_val = pd.read_csv("player_valuations.csv")
df_val["date"] = pd.to_datetime(df_val["date"])
df_val["year"] = df_val["date"].dt.year

# only take the final valuation of each player for each year
# this is to ensure we have the most recent valuation for each player in that year
val_latest_year = df_val.sort_values("date").drop_duplicates(["player_id", "year"], keep="last")

val_yearly = val_latest_year[["player_id", "year", "market_value_in_eur"]]

# Restructuring data from transfers.csv
df_transfers = pd.read_csv("transfers.csv")
df_transfers["transfer_date"] = pd.to_datetime(df_transfers["transfer_date"])
df_transfers["year"] = df_transfers["transfer_date"].dt.year

# If there was a transfer, transfer fee. Else, NaN
transfers_yearly = df_transfers[["player_id", "year", "transfer_fee"]]

# Merging all dataframes into a single dataframe
merged_df = player_yearly.merge(val_yearly, on=["player_id", "year"], how="left")
merged_df = merged_df.merge(transfers_yearly, on=["player_id", "year"], how="left")

df_players = pd.read_csv("players.csv")

# Adding player information to the merged dataframe
panel_df = merged_df.merge(
    df_players[["player_id", "position", "foot", "height_in_cm", "country_of_citizenship"]],
    on="player_id", how="left"
)

panel_df.to_csv("panel_df.csv", index=False)
print("Merged panel_df saved to 'panel_df.csv'.")


