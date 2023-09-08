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

    df = df[PLAYER_PROCESSING_COLUMNS]

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
    fifty = batsman_score_game[batsman_score_game['batsman_run'] >= 50]\
        .groupby(['batter'], as_index=False).count().rename(columns={'batsman_run': '50s'})
    hundred = batsman_score_game[batsman_score_game['batsman_run'] >= 100]\
        .groupby(['batter'], as_index=False).count().rename(columns={'batsman_run': '100s'})
    merged_totals = fifty.merge(hundred, on='batter', how='outer').fillna(0)
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

    batting_stats_merged = batting_stats_merged.rename(columns={'batter': 'player'})

    print("✅ Batting Features Created")

    return batting_stats_merged


def player_bowling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the ball by ball match cleaned dataframe as a base.
    Creates a dataframe of player bowling features extracted from the
    ball by ball match dataframe.
    Returns the features processed.
    """
    assert isinstance(df, pd.DataFrame)

    df = df[PLAYER_PROCESSING_COLUMNS]

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

    bowling_stats_merged = bowling_stats_merged.rename(columns={'bowler': 'player'})

    print("✅ Bowling Features Created")

    return bowling_stats_merged


def win_loss_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a cleaned dataframe as a base.
    Creates a dataframe of player win loss statistics.
    Returns the features processed.
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df['Team1Players'][0], list)

    df = df[WIN_LOSS_PROCESSING_COLUMNS]

    ### Group to remove duplicates
    df = df.groupby('ID', as_index=False).first()

    ### Team 1 Wins
    def team_1_wins(Team1, Team2, WinningTeam):
        if WinningTeam == Team1:
            return 1
        elif WinningTeam == Team2:
            return 0
        else:
            return -1

    df['team_1_win'] = df.apply(lambda row: \
        team_1_wins(row['Team1'], row['Team2'], row['WinningTeam']), axis=1)

    ### Winning Player Count
    def winning_team_players(team_1_win, Team1Players, Team2Players):
        if team_1_win == 1:
            return Team1Players
        elif team_1_win == 0:
            return Team2Players
        else:
            return []

    df['winning_team_players'] = df.apply(lambda row: winning_team_players\
        (row['team_1_win'], row['Team1Players'], row['Team2Players']), axis=1)

    player_wins = []
    def player_wins_list(winning_team_players):
        for row in winning_team_players:
            for player in row:
                player_wins.append(player)
        return player_wins

    player_wins_list(df['winning_team_players'])

    player_win_count = {}
    for player in player_wins:
        if player in player_win_count:
            player_win_count[player] += 1
        else:
            player_win_count[player] = 1

    wins = pd.DataFrame({'player': player_win_count.keys(), \
                        'wins': player_win_count.values()})

    ### Losing Player Count
    def losing_team_players(team_1_win, Team1Players, Team2Players):
        if team_1_win == 0:
            return Team1Players
        elif team_1_win == 1:
            return Team2Players
        else:
            return []

    df['losing_team_players'] = df.apply(lambda row: losing_team_players\
        (row['team_1_win'], row['Team1Players'], row['Team2Players']), axis=1)

    player_loses = []
    def player_loses_list(losing_team_players):
        for row in losing_team_players:
            for player in row:
                player_loses.append(player)
        return player_loses

    player_loses_list(df['losing_team_players'])

    player_loss_count = {}
    for player in player_loses:
        if player in player_loss_count:
            player_loss_count[player] += 1
        else:
            player_loss_count[player] = 1

    loses = pd.DataFrame({'player': player_loss_count.keys(), \
                            'loses': player_loss_count.values()})

    ### Matches
    matches = {}

    for player in player_win_count:
        if player in player_loss_count:
            matches[player] = player_win_count[player] + player_loss_count[player]
        elif player not in player_loss_count:
            matches[player] = player_win_count[player]

    for player in player_loss_count:
        if player not in player_win_count:
            matches[player] = player_loss_count[player]

    matches = pd.DataFrame({'player': matches.keys(), \
                            'matches': matches.values()})

    ### Merge
    win_loss_merge = matches.merge(wins, on='player', how='outer')\
                        .merge(loses, on='player', how='outer').fillna(0)

    ### Create Win Ratio
    win_loss_merge['win_ratio'] = \
        round((win_loss_merge['wins']/win_loss_merge['matches']), 2)

    print("✅ Win Loss Features Created")

    return win_loss_merge


def player_features_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a cleaned dataframe as a base.
    Compiles the final player dataset.
    """
    assert isinstance(df, pd.DataFrame)

    batting = player_batting_features(df)
    bowling = player_bowling_features(df)
    win_loss = win_loss_features(df)

    final_player_dataset = batting.merge(bowling, on='player', how='outer')\
                            .merge(win_loss, on='player', how='outer').fillna(0)

    print("✅ Final Player Dataset Created")

    return final_player_dataset
