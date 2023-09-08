import glob
import os
import time
import pickle
import xgboost as xgb

from google.cloud import storage
from model.params import *

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
    '''
    This will save the model locally at f"{LOCAL_REGISTRY} and
    also save the model on GCS in your bucket on GCS at "models/{timestamp}.h5"
    '''

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save the model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.model")
    model.save_model(model_path)

    print("✅ Model saved locally")

    # Save the model to GCS

    model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None

def load_model() -> xgb.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    if MODEL_TARGET == "local":
        print(f"\nLoad latest model from local registry...")

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(f"\nLoad latest model from disk...")

        latest_model = xgb.Booster(model_file=most_recent_model_path_on_disk)

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
