import numpy as np
import pandas as pd

from model.params import *

def player_batting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the ball by ball match cleaned dataframe as a base.
    Creates a dataframe of player batting features extracted from the
    ball by ball match dataframe.
    Returns the features processed.
    """
    assert isinstance(df, pd.DataFrame)

    ### Total Runs
    batter_total_runs = df.groupby(['batter'], as_index=False)['batsman_run']\
        .sum('batsman_run').rename(columns={'batsman_run': 'total_runs'})

    ### Balls Faced
    balls_faced = df[df['extra_type'].isin(['noextra', 'legbyes', 'byes', 'noballs'])]\
                .groupby(['batter'], as_index=False)['ballnumber'].count()\
                .rename(columns={'ballnumber': 'balls_faced'})

    ### Batter Innings
    batter_innings = df.groupby(['batter'], as_index=False)['ID'].nunique()\
                    .rename(columns={'ID': 'batter_innings'})
    non_striker_innings = df.groupby(['non_striker'], as_index=False)['ID'].nunique()\
                    .rename(columns={'ID': 'non_striker_innings', 'non_striker': 'batter'})
    combined_innings_df = batter_innings.merge(non_striker_innings, \
                    on='batter', how='outer').fillna(0)
    combined_innings_df['bat_innings'] = combined_innings_df[['batter_innings', 'non_striker_innings']].max(axis=1)
    bat_innings = combined_innings_df.drop(columns=['batter_innings', 'non_striker_innings'])

    ### Boundaries
    non_boundary = list(df[df['non_boundary'] == 1].index)
    temp_df = df.drop(index=non_boundary)
    fours = temp_df[temp_df['batsman_run'] == 4].groupby(['batter'], as_index=False)\
            ['batsman_run'].count().rename(columns={'batsman_run': 'fours'})
    sixes = temp_df[temp_df['batsman_run'] == 6].groupby(['batter'], as_index=False)\
            ['batsman_run'].count().rename(columns={'batsman_run': 'sixes'})

    ### Batsman Totals
    batsman_score_game = df.groupby(['ID', 'batter'], as_index=False)\
                        ['batsman_run'].sum().drop(columns='ID')
    zero = batsman_score_game[batsman_score_game['batsman_run'] == 0]\
        .groupby(['batter'], as_index=False).count().rename(columns={'batsman_run': 'zero'})
    fifty = batsman_score_game[batsman_score_game['batsman_run'] >= 50]\
        .groupby(['batter'], as_index=False).count().rename(columns={'batsman_run': '50s'})
    hundred = batsman_score_game[batsman_score_game['batsman_run'] >= 100]\
        .groupby(['batter'], as_index=False).count().rename(columns={'batsman_run': '100s'})
    merged_totals = zero.merge(fifty, on='batter', how='outer').merge(hundred, on='batter', how='outer').fillna(0)
    merged_totals['zero'] = merged_totals['zero'].astype('int')
    merged_totals['50s'] = merged_totals['50s'].sub(merged_totals['100s']).astype('int')
    merged_totals['100s'] = merged_totals['100s'].astype('int')

    ### Batsman Not Out
    last_ball_index_temp = list(df.groupby(['ID', 'innings'], as_index=False)['innings']\
                        .idxmax().sort_values(by='innings', ascending=False)['innings'])
    last_ball_index_temp.remove(0)
    last_ball_index = [index - 1 for index in last_ball_index_temp]
    last_ball_index.insert(0, df.shape[0])
    player_out_final_ball = df[(df['isWicketDelivery']==1) & (df.index.isin(last_ball_index))]\
                            .groupby(['player_out'], as_index=False)['ID'].count()\
                            .rename(columns={'ID': 'out_count', 'player_out': 'batter'})
    final_ball_batter = df[df.index.isin(last_ball_index)].groupby(['batter'], as_index=False)\
                        ['ID'].count().rename(columns={'ID': 'at_bat_count'})
    final_ball_non_striker = df[df.index.isin(last_ball_index)].groupby(['non_striker'], as_index=False)\
                            ['ID'].count().rename(columns={'ID': 'non_striker_count', 'non_striker': 'batter'})
    combined_final_ball = final_ball_batter.merge(final_ball_non_striker, on='batter', how='outer')\
                            .merge(player_out_final_ball, on='batter', how='outer')
    combined_final_ball.fillna(0, inplace=True)
    combined_final_ball['not_out'] = combined_final_ball['at_bat_count']\
                                    .add(combined_final_ball['non_striker_count'])\
                                    .sub(combined_final_ball['out_count']).astype('int')
    not_out = combined_final_ball[['batter', 'not_out']]

    ### High Score
    high_score = df.groupby(['ID', 'batter'], as_index=False)['batsman_run']\
                        .sum().groupby('batter', as_index=False)['batsman_run'].max()\
                        .rename(columns={'batsman_run': 'high_score'})

    ### Merge
    batting_stats_merged = bat_innings.merge(batter_total_runs, on='batter', how='outer')\
                        .merge(not_out, on='batter', how='outer')\
                        .merge(merged_totals, on='batter', how='outer')\
                        .merge(fours, on='batter', how='outer')\
                        .merge(sixes, on='batter', how='outer')\
                        .merge(high_score, on='batter', how='outer')\
                        .merge(balls_faced, on='batter', how='outer').fillna(0)

    ### Feature creation
    batting_stats_merged['average_score'] = round\
        ((batting_stats_merged['total_runs'] / batting_stats_merged['bat_innings']), 2).fillna(0)
    batting_stats_merged['batting_average'] = round((batting_stats_merged['total_runs'] / \
        (batting_stats_merged['bat_innings'] - batting_stats_merged['not_out'])), 2).fillna(0)
    batting_stats_merged['batting_average'] = np.where(batting_stats_merged\
        ['batting_average'] == np.inf, batting_stats_merged['average_score'], batting_stats_merged['batting_average'])
    batting_stats_merged['batting_strike_rate'] = round\
        ((batting_stats_merged['total_runs'] * 100)/batting_stats_merged['balls_faced'], 2).fillna(0)

    print("✅ Batting Features Created")

    return batting_stats_merged


def player_batting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the ball by ball match cleaned dataframe as a base.
    Creates a dataframe of player bowling features extracted from the
    ball by ball match dataframe.
    Returns the features processed.
    """
    assert isinstance(df, pd.DataFrame)

    ### Bowler Innings
    bowler_innings = df.groupby(['bowler'], as_index=False)['ID'].nunique()\
                    .rename(columns={'ID': 'bowler_innings'})

    ### Balls Bowled
    balls_bowled = df[df['extra_type'].isin(['noextra', 'legbyes', 'byes', 'penalty'])]\
                .groupby(['bowler'], as_index=False)['ballnumber'].count()\
                .rename(columns={'ballnumber': 'balls_bowled'})

    ### Runs Conceded
    runs_conceded = df[df['extra_type'].isin(['noextra', 'wides', 'noballs'])]\
                .groupby(['bowler'], as_index=False)['total_run'].sum()\
                .rename(columns={'total_run': 'runs_conceded'})

    ### Wickets Taken
    wickets = df[(df['isWicketDelivery'] == 1) & (df['kind'].isin\
                (['nokind', 'caught', 'caught and bowled', 'bowled','stumped', 'lbw', 'hit wicket']))]\
                .groupby(['bowler'], as_index=False)['isWicketDelivery'].count()\
                .rename(columns={'isWicketDelivery': 'wickets'})

    ### Wickets Per Game Stats
    wicket_count_per_game = df[(df['isWicketDelivery'] == 1) & (df['kind'].isin\
                (['nokind', 'caught', 'caught and bowled', 'bowled','stumped', 'lbw', 'hit wicket']))]\
                .groupby(['ID', 'bowler'], as_index=False)['isWicketDelivery'].count()\
                .rename(columns={'isWicketDelivery': 'wickets'})

    four_wickets = wicket_count_per_game[wicket_count_per_game['wickets'] == 4]\
                .groupby(['bowler'], as_index=False)['wickets'].count()\
                .rename(columns={'wickets': 'four_wickets'})

    five_wickets = wicket_count_per_game[wicket_count_per_game['wickets'] == 5]\
                .groupby(['bowler'], as_index=False)['wickets'].count()\
                .rename(columns={'wickets': 'five_wickets'})

    ### Merge
    bowling_stats_merged = bowler_innings.merge(balls_bowled, on='bowler', how='outer')\
                        .merge(runs_conceded, on='bowler', how='outer')\
                        .merge(wickets, on='bowler', how='outer')\
                        .merge(four_wickets, on='bowler', how='outer')\
                        .merge(five_wickets, on='bowler', how='outer').fillna(0)

    ### Feature Creation
    bowling_stats_merged['bowling_average'] = round\
        ((bowling_stats_merged['runs_conceded'] / bowling_stats_merged['wickets']), 2).fillna(0)
    bowling_stats_merged['bowling_average'] = np.where\
        (bowling_stats_merged['bowling_average'] == np.inf, 0, bowling_stats_merged['bowling_average'])
    bowling_stats_merged['bowling_economy_rate'] = round\
        ((bowling_stats_merged['runs_conceded']/(bowling_stats_merged['balls_bowled'] / 6)), 2)
    bowling_stats_merged['bowling_strike_rate'] = round((bowling_stats_merged['balls_bowled']/bowling_stats_merged['wickets']), 2)
    bowling_stats_merged['bowling_strike_rate'] = np.where\
        (bowling_stats_merged['bowling_strike_rate'] == np.inf, 0, bowling_stats_merged['bowling_strike_rate'])

    print("✅ Bowling Features Created")

    return bowling_stats_merged
