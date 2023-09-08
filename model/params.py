import os

#####CONSTANTS#####
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "ipl_prediction_model", "raw_data")
LOCAL_MODELS_PATH =  os.path.join(os.path.expanduser('~'), "code", "ipl_prediction_model", "model", "trained_models")

##################  VARIABLES  ##################
BUCKET_NAME = os.environ.get("BUCKET_NAME") # Make sure the env file is made
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
MODEL_TARGET = os.environ.get("MODEL_TARGET")


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

CATAGORICAL_COLUMNS = ['Team1', 'Team2', 'Venue', 'TossDecision']

NUMERICAL_COLUMNS = ['Team2_innings_total', 'Team_1_batting_average',
                 'Team_2_batting_average', 'team_1_toss_winner', 'MatchImportance',
                 'Team1_points_against_avg', 'Team2_points_against_avg',
                 'Team1_MVP_average', 'Team2_MVP_average']

##################  CONSTANTS  ##################
PLAYER_PROCESSING_COLUMNS = ['ID', 'innings', 'overs', 'ballnumber', 'batter',
        'bowler', 'non_striker', 'extra_type', 'batsman_run', 'extras_run',
        'total_run', 'non_boundary', 'isWicketDelivery', 'player_out', 'kind',
        'fielders_involved', 'BattingTeam']

WIN_LOSS_PROCESSING_COLUMNS = ['ID', 'Season', 'Team1', 'Team2', 'WinningTeam',
                             'Player_of_Match', 'Team1Players', 'Team2Players']
