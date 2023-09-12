import os

#####CONSTANTS#####
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "ipl_prediction_model", "raw_data")
LOCAL_MODELS_PATH =  os.path.join(os.getcwd(), "ipl_model", "model")
LOCAL_PREPROCESSORS_PATH = os.path.join(os.getcwd(), "ipl_model", "preprocessor")
DOCKER_MODEL_PATH = os.path.join(os.path.expanduser('~'), "ipl_model", "model")
DOCKER_PREPROCESSOR_PATH = os.path.join(os.path.expanduser('~'), "ipl_model", "preprocessor")

##################  VARIABLES  ##################
BUCKET_NAME = os.environ.get("BUCKET_NAME") # Make sure the env file is made
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")
MODEL_TARGET = os.environ.get("MODEL_TARGET")

##################  CONSTANTS  ##################
PLAYER_PROCESSING_COLUMNS = ['ID', 'innings', 'overs', 'ballnumber', 'batter',
        'bowler', 'non_striker', 'extra_type', 'batsman_run', 'extras_run',
        'total_run', 'non_boundary', 'isWicketDelivery', 'player_out', 'kind',
        'fielders_involved', 'BattingTeam']

WIN_LOSS_PROCESSING_COLUMNS = ['ID', 'Season', 'Team1', 'Team2', 'WinningTeam',
                             'Player_of_Match', 'Team1Players', 'Team2Players']

COLUMN_NAMES = ['City', 'Team1', 'Team2', 'Venue', 'TossWinner',
       'TossDecision', 'WinningTeam', 'Avg_Weighted_Score_diff',
       'batting_average_weighted_diff', 'batting_strike_rate_weighted_diff',
       'bowling_average_diff', 'bowling_economy_rate_diff', 'win_ratio_diff']

TARGET_NAME = ['WinningTeam']

FEATURE_NAMES = ['City', 'Team1', 'Team2', 'Venue', 'TossWinner',
       'TossDecision', 'Avg_Weighted_Score_diff',
       'batting_average_weighted_diff', 'batting_strike_rate_weighted_diff',
       'bowling_average_diff', 'bowling_economy_rate_diff', 'win_ratio_diff']

CATAGORICAL_COLUMNS = ['Team1', 'Team2', 'Venue', 'TossDecision', 'City']

NUMERICAL_COLUMNS = ['TossWinner', 'Avg_Weighted_Score_diff',
       'batting_average_weighted_diff', 'batting_strike_rate_weighted_diff',
       'bowling_average_diff', 'bowling_economy_rate_diff', 'win_ratio_diff']

TEAM_NAMES = ['rajasthan royals', 'royal challengers bangalore',
       'sunrisers hyderabad', 'delhi capitals', 'chennai super kings',
       'gujarat titans', 'lucknow super giants', 'kolkata knight riders',
       'punjab kings', 'mumbai indians', 'rising pune supergiant',
       'gujarat lions', 'pune warriors', 'deccan chargers',
       'kochi tuskers kerala']

##################  CONSTANTS  ##################
PLAYER_PROCESSING_COLUMNS = ['ID', 'innings', 'overs', 'ballnumber', 'batter',
        'bowler', 'non_striker', 'extra_type', 'batsman_run', 'extras_run',
        'total_run', 'non_boundary', 'isWicketDelivery', 'player_out', 'kind',
        'fielders_involved', 'BattingTeam']

WIN_LOSS_PROCESSING_COLUMNS = ['ID', 'Season', 'Team1', 'Team2', 'WinningTeam',
                             'Player_of_Match', 'Team1Players', 'Team2Players']
