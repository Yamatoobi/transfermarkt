# This script restructures and merges football player data from multiple CSV files
import pandas as pd

# ==============Restructuring data from Appearances.csv==============
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

# ==============Add club info from Appearances.csv==============
# For each player-year, get the most frequent club (in case of mid-season transfer)
club_mode = df_app.groupby(["player_id", "year"])["player_club_id"].agg(
    lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
).reset_index()
club_mode = club_mode.rename(columns={"player_club_id": "club_id"})

player_yearly = player_yearly.merge(club_mode, on=["player_id", "year"], how="left")

# ==============Restructuring data from player_valuations.csv==========
df_val = pd.read_csv("player_valuations.csv")
df_val["date"] = pd.to_datetime(df_val["date"])
df_val["year"] = df_val["date"].dt.year

# only take the final valuation of each player for each year
val_latest_year = df_val.sort_values("date").drop_duplicates(["player_id", "year"], keep="last")
val_yearly = val_latest_year[["player_id", "year", "market_value_in_eur"]]

# ===============Restructuring data from transfers.csv============
df_transfers = pd.read_csv("transfers.csv")
df_transfers["transfer_date"] = pd.to_datetime(df_transfers["transfer_date"])
df_transfers["year"] = df_transfers["transfer_date"].dt.year

# If there was a transfer, transfer fee. Else, NaN
transfers_yearly = df_transfers[["player_id", "year", "transfer_fee"]]

# Merging all dataframes into a single dataframe
merged_df = player_yearly.merge(val_yearly, on=["player_id", "year"], how="left")
merged_df = merged_df.merge(transfers_yearly, on=["player_id", "year"], how="left")

df_players = pd.read_csv("players.csv")
df_players["date_of_birth"] = pd.to_datetime(df_players["date_of_birth"], errors="coerce")

# =============Add starter vs substitute information from game_lineups.csv==========
df_lineups = pd.read_csv("game_lineups.csv")
df_lineups["date"] = pd.to_datetime(df_lineups["date"])
df_lineups["year"] = df_lineups["date"].dt.year

# Count number of starts and subs by player-year
lineup_stats = df_lineups.groupby(["player_id", "year", "type"]).size().unstack(fill_value=0).reset_index()
lineup_stats = lineup_stats.rename(columns={"starting_lineup": "starts", "substitutes": "subs"})

# ============Add defensive contributions from game_events.csv=================
df_events = pd.read_csv("game_events.csv")
df_events["date"] = pd.to_datetime(df_events["date"])
df_events["year"] = df_events["date"].dt.year

# List of defensive keywords to count separately
def_keywords = ["tackle", "interception", "block", "clearance"]

# For each keyword, create a column with the count per player-year
for keyword in def_keywords:
    col_name = f"{keyword}s"  # e.g., "tackles"
    df_events[col_name] = df_events["description"].str.lower().fillna("").str.contains(keyword).astype(int)

# Group by player_id and year, summing each defensive action
defense_stats = df_events.groupby(["player_id", "year"])[[f"{k}s" for k in def_keywords]].sum().reset_index()

# ============Add club information from clubs.csv=================
df_clubs = pd.read_csv("clubs.csv")
# Merge club name and total_market_value into merged_df using club_id
merged_df = merged_df.merge(df_clubs[["club_id", "name", "total_market_value"]], on="club_id", how="left")
# 'name' is club name, 'total_market_value' is club's market value

# ==========Merge all stats into panel_df================
# Adding player information to the merged dataframe
panel_df = merged_df.merge(
    df_players[["player_id", "sub_position", "foot", "height_in_cm", "country_of_citizenship", "date_of_birth"]],
    on="player_id", how="left"
)

# Use sub_position instead of position for clarity
panel_df = panel_df.rename(columns={"sub_position": "position"})

panel_df["age"] = panel_df.apply(
    lambda row: row["year"] - row["date_of_birth"].year if pd.notnull(row["date_of_birth"]) else None,
    axis=1
)

panel_df = panel_df.merge(lineup_stats[["player_id", "year", "starts", "subs"]], on=["player_id", "year"], how="left")
panel_df = panel_df.merge(defense_stats, on=["player_id", "year"], how="left")

panel_df.to_csv("panel_df.csv", index=False)
print("Merged panel_df saved to 'panel_df.csv'.")