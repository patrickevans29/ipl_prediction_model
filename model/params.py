##################  VARIABLES  ##################
BUCKET_NAME = os.environ.get("BUCKET_NAME") # Make sure the env file is made
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")

COLUMN_NAMES_RAW = []

##################  CONSTANTS  ##################
PLAYER_PROCESSING_COLUMNS = ['ID', 'innings', 'overs', 'ballnumber', 'batter',
        'bowler', 'non_striker', 'extra_type', 'batsman_run', 'extras_run',
        'total_run', 'non_boundary', 'isWicketDelivery', 'player_out', 'kind',
        'fielders_involved', 'BattingTeam']

WIN_LOSS_PROCESSING_COLUMNS = ['ID', 'Season', 'Team1', 'Team2', 'WinningTeam',
                             'Player_of_Match', 'Team1Players', 'Team2Players']
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".iplpredictionmodel", "model", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".iplpredictionmodel", "model", "training_outputs")