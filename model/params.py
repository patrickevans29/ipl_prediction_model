#####CONSTANTS#####
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".iplpredictionmodel", "model", "data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".iplpredictionmodel", "model", "training_outputs")

#####VARIABLES#####
BUCKET_NAME = os.environ.get("BUCKET_NAME") # Make sure the env file is made
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BQ_DATASET = os.environ.get("BQ_DATASET")

COLUMN_NAMES_RAW = []
