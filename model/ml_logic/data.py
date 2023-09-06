import pandas as pd

'''
This module is used for cleaning data that can be
be used to pass into the model in the model.py function
'''

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean data by:
    - Removing or imputing NaN
    '''
    pass # ADD CODE HERE

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This is where we create the new features for the model.
    They should be:
    - team_1_win --> 1 or 0 if team 1 one the match
    - innings_total --> to be used to calculate batting avgs etc.
    - team_1_batting_average --> the average amount of runs that team1 scores in a match
    - team_2_batting_average --> the average amount of runs that team2 scores in a match
    - team_1_toss_winner --> 1 or 0 if team1 won the toss
    - match_importance --> ordinal value given to the importance of the match. Final or regular etc.
    - team_1_points_against_avg --> the average number of point opposition teams score agains team_1
    - team_2_points_against_avg --> the average number of point opposition teams score agains team_2
    - team_1_avg_mvp --> average number of times that team_1 has MVP
    - team_2_avg_mvp --> average number of times that team_2 has MVP
    '''
    df = clean_data(df)
    pass # ADD CODE HERE
