import numpy as np
import pandas as pd
from pathlib import Path

from ipl_model.params import *
from ipl_model.ml_logic.feature_engineer import player_features_dataset

def match_filter(df: pd.DataFrame,
               number_of_matches: int = "all",
               team_1: list = "all",
               team_2: list = "all",
               season: list = "all") -> pd.DataFrame:
    """
    Takes a cleaned dataframe as a base.
    Filters the dataframe by number of targets:

    number_of_matches, the maximum number of matches
    team_1, the list of teams to include as team_1
    team_2, the list of teams to include as team_2
    season, the list of seasons to include in the filter.

    Returns a filtered dataframe
    """

    if team_1 != "all":
        df = df[df['Team1'].isin(team_1)]

    if team_2 != "all":
        df = df[df['Team2'].isin(team_2)]

    if season != "all":
        df = df[df['Season'].isin(season)]

    if number_of_matches != "all":
        match_id = df['ID'].unique().tolist()
        df = df[df['ID'].isin(match_id[0:number_of_matches])]

    return df


def player_team_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a cleaned dataframe as a base.

    Processes the dataframe to output player data using the feature engineer
    module

    Returns a dataframe for both team_1 and team_2 players
    """

    match_id = df['ID'].unique().tolist()

    team_1_list = []
    team_2_list = []

    for match in match_id:
        match_df = df[df['ID']==match]

        team_1_players = match_df['Team1Players'].iloc[0]
        team_2_players = match_df['Team2Players'].iloc[0]

        player_match_df = player_features_dataset(match_df)

        t1 = player_match_df[player_match_df['player'].isin(team_1_players)]
        t2 = player_match_df[player_match_df['player'].isin(team_2_players)]

        t1.loc[:, 'ID'] = match
        t2.loc[:, 'ID'] = match

        team_1_list.append(t1)
        team_2_list.append(t2)

    team_1 = pd.concat(team_1_list, ignore_index=True)
    team_2 = pd.concat(team_2_list, ignore_index=True)

    return team_1, team_2


def team_average(df_filtered: pd.DataFrame,
                 team_1: pd.DataFrame,
                 team_2: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the filtered dataframe from match_filter()
    and team_1, team_2 dataframe from player_team_data()

    Processes the dataframe to output a combined dataframe
    with average scores for each team grouped by the match ID

    Returns a combined dataframe.
    """

    df = df_filtered[WIN_LOSS_PROCESSING_COLUMNS].groupby('ID').first()
    columns = team_1.drop(columns='player').columns

    for column in columns:
        df[f"Team_1_{column}"] = team_1.groupby('ID')[column].mean()
        df[f"Team_2_{column}"] = team_2.groupby('ID')[column].mean()

    return df


def game_average(df: pd.DataFrame,
                number_of_matches: int = "all",
                team_1: list = "all",
                team_2: list = "all",
                season: list = "all") -> pd.DataFrame:
    """
    Takes a cleaned dataframe as a base.

    A combined process for returning the average match data for a filtered
    dataframe using:
    match_filter()
    player_team_data()
    team_average()

    number_of_matches, the maximum number of matches
    team_1, the list of teams to include as team_1
    team_2, the list of teams to include as team_2
    season, the list of seasons to include in the filter.

    Returns a dataframe
    """

    df_filtered = match_filter(df, number_of_matches, team_1, team_2, season)

    team_1_df, team_2_df = player_team_data(df_filtered)

    average_df = team_average(df_filtered, team_1_df, team_2_df)

    return average_df
