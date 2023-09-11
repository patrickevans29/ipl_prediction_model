import glob
import os
import time
import pickle
import xgboost as xgb
import pandas as pd

from typing import Any
from google.cloud import storage

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

def save_model(model) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.xgbmodel"
    model_path = os.path.join(LOCAL_MODELS_PATH, model_filename)

    # Create the directory if it doesn't exist
    os.makedirs(LOCAL_MODELS_PATH, exist_ok=True)

    # Save the model locally using pickle
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

    print("✅ XGBoost model saved locally")

    # Save the model to GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None

def load_model() -> Any:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only

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
        print(f"\nLoad latest XGBoost model from GCS...")

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models/"))


        # Find the blob with the latest timestamp
        latest_blob = max(blobs, key=lambda x: x.updated)

        # Specify the path to save the downloaded model
        latest_model_path_to_save = os.path.join(LOCAL_MODELS_PATH, os.path.basename(latest_blob.name))

        # Download the latest model from GCS
        latest_blob.download_to_filename(latest_model_path_to_save)

        # Load the downloaded XGBoost model
        #latest_model = xgb.Booster(model_file=latest_model_path_to_save)
        with open(latest_model_path_to_save, 'rb') as model_file:
                    latest_model = pickle.load(model_file)


        print(f"✅ Latest model downloaded from cloud storage @ {latest_model_path_to_save}")

        return latest_model

        """except:
            print(f"\n❌ No XGBoost model found in GCS bucket {BUCKET_NAME} @ {latest_model_path_to_save}")

            return None"""

def load_preprocessor() -> Any:
    """
    Load a saved preprocessor:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'

    Return None (but do not Raise) if no preprocessor is found

    """

    if MODEL_TARGET == "local":
        print(f"\nLoad latest preprocessor from local registry...")

        # Get a list of all preprocessor paths in the specified directory
        local_preprocessor_paths = glob.glob(f"{LOCAL_PREPROCESSORS_PATH}/*")

        if not local_preprocessor_paths:
            print(f"No preprocessor files found in the local directory: {LOCAL_PREPROCESSORS_PATH}")
            return None

        # Sort the preprocessor paths by filename (which contains the timestamp)
        local_preprocessor_paths_sorted = sorted(local_preprocessor_paths)

        # Choose the most recent preprocessor path (last in the sorted list)
        most_recent_preprocessor_path_on_disk = local_preprocessor_paths_sorted[-1]

        print(f"\nLoad latest preprocessor from disk...")

        # Load the most recent preprocessor using pickle
        with open(most_recent_preprocessor_path_on_disk, 'rb') as preprocessor_file:
            latest_preprocessor = pickle.load(preprocessor_file)

        print("✅ Preprocessor loaded from local disk")

        return latest_preprocessor

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(f"\nLoad latest preprocessor from GCS...")

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="preprocessors/"))

        try:
            # Find the blob with the latest timestamp
            latest_blob = max(blobs, key=lambda x: x.updated)

            # Specify the path to save the downloaded preprocessor
            latest_preprocessor_path_to_save = os.path.join(LOCAL_PREPROCESSORS_PATH, os.path.basename(latest_blob.name))

            # Download the latest preprocessor from GCS
            latest_blob.download_to_filename(latest_preprocessor_path_to_save)

            # Load the downloaded preprocessor using pickle
            with open(latest_preprocessor_path_to_save, 'rb') as preprocessor_file:
                latest_preprocessor = pickle.load(preprocessor_file)

            print(f"✅ Latest preprocessor downloaded from cloud storage @ {latest_preprocessor_path_to_save}")

            return latest_preprocessor

        except:
            print(f"\n❌ No preprocessor found in GCS bucket {BUCKET_NAME} @ {latest_preprocessor_path_to_save}")

            return None
