import os


#####CONSTANTS#####
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".iplpredictionmodel", "model", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".iplpredictionmodel", "model", "training_outputs")

#####VARIABLES#####
BUCKET_NAME = os.environ.get("BUCKET_NAME") # Make sure the env file is made
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")

COLUMN_NAMES = ['Team1', 'Team2', 'Venue', 'TossDecision',
                'Team1_Win', 'Team2_innings_total', 'Team_1_batting_average',
                'Team_2_batting_average', 'team_1_toss_winner', 'MatchImportance',
                'Team1_points_against_avg', 'Team2_points_against_avg',
                'Team1_MVP_average', 'Team2_MVP_average']
TARGET_NAME = ['Team1_Win']
FEATURE_NAMES = ['Team1', 'Team2', 'Venue', 'TossDecision',
                 'Team2_innings_total', 'Team_1_batting_average',
                 'Team_2_batting_average', 'team_1_toss_winner', 'MatchImportance',
                 'Team1_points_against_avg', 'Team2_points_against_avg',
                 'Team1_MVP_average', 'Team2_MVP_average']
