import glob
import os
import time
import pickle
import xgboost as xgb
import pandas as pd

from typing import Any

#from google.cloud import storage
from ipl_model.params import *

def load_data_locally() -> pd.DataFrame:
    '''
    Accesses the raw_data directory to retrieve the two datasets
    and merges them in a pandas dataframe. This data can then be
    cleaned with clean_data().
    '''
    # Create the file path to the ball_by_ball.csv file
    csv_file_path_ball = os.path.join(LOCAL_DATA_PATH, "IPL_Ball_by_Ball_2008_2022.csv")
    print(csv_file_path_ball)

    # Check to see if the file is there
    if os.path.exists(csv_file_path_ball):
        # Load it into a pandas df
        ball_by_ball_df = pd.read_csv(csv_file_path_ball)
    else:
        print(f"The file {csv_file_path_ball} does not exist at {LOCAL_DATA_PATH}.")

    # Create the file path to the IPL_Matches_2008_2022 file
    csv_file_path_match = os.path.join(LOCAL_DATA_PATH, "IPL_Matches_2008_2022.csv")

    # Check to see if the file is there
    if os.path.exists(csv_file_path_match):
        # Load it into a pandas df
        match_df = pd.read_csv(csv_file_path_match)
    else:
        print(f"The file {csv_file_path_match} does not exist at {LOCAL_DATA_PATH}.")

    # Merge the two dataframes based on the "ID" column
    merged_df = pd.merge(ball_by_ball_df, match_df, on="ID", how="inner")  # Change how to "inner" if you want to keep only matching rows

    return merged_df

def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save params locally
    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", timestamp + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

def save_model(model) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(LOCAL_MODELS_PATH, f"{timestamp}.xgbmodel")

    # Create the directory if it doesn't exist
    os.makedirs(LOCAL_MODELS_PATH, exist_ok=True)

    # Save the model locally using pickle
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    print("✅ XGBoost model saved locally")

    return None

    # Save the model to GCS
    '''
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")'''

    return None

def load_model() -> Any:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(f"\nLoad latest model from local registry...")

        # Get a list of all model paths in the specified directory
        local_model_paths = glob.glob(f"{LOCAL_MODELS_PATH}/*")

        if not local_model_paths:
            print(f"No model files found in the local directory: {local_model_paths}")
            return None

        # Sort the model paths by filename (which contains the timestamp)
        local_model_paths_sorted = sorted(local_model_paths)

        # Choose the most recent model path (last in the sorted list)
        most_recent_model_path_on_disk = local_model_paths_sorted[-1]

        print(f"\nLoad latest model from disk...")

        # Load the most recent model using pickle
        with open(most_recent_model_path_on_disk, 'rb') as model_file:
            latest_model = pickle.load(model_file)

        print("✅ Model loaded from local disk")

        return latest_model

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(f"\nLoad latest model from GCS...")

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

        try:
            # Find the blob with the latest timestamp
            latest_blob = max(blobs, key=lambda x: x.updated)

            # Specify the path to save the downloaded model
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)

            # Download the latest model from GCS
            latest_blob.download_to_filename(latest_model_path_to_save)

            # Load the downloaded XGBoost model
            latest_model = xgb.Booster(model_file=latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model

        except:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None
