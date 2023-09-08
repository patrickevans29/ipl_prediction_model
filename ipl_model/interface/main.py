import numpy as np
import pandas as pd

from pathlib import Path
from dateutil.parser import parse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ipl_model.params import *
from ipl_model.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from ipl_model.ml_logic.make_model import initialize_model, train_model
from ipl_model.ml_logic.preprocessor import preprocess_features
from ipl_model.ml_logic.registry import load_model, save_model, save_results

def preprocess() -> None:
    """
    - Query the raw dataset from our BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print("⭐️ preprocess")

    query = f"""
        ### ALL THIS WILL NEED TO BE CHANGED ###
        SELECT {",".join(COLUMN_NAMES_RAW)} # These are the column names that we need to get
        FROM {GCP_PROJECT}.{BQ_DATASET}.raw_{DATA_SIZE}
        ORDER BY # If necessary
    """

    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query.csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )

    # Process data
    data_clean = clean_data(data_query) # data_query needs to be a .csv file

    ###################### IMPORTANT #########################
    # Select the columns that we need to put into our model
    X = data_clean.drop("team1_win", axis=1)
    y = data_clean[["team1_win"]]

    X_processed = preprocess_features(X)

    # Load a DataFrame onto BigQuery containing [ID, X_processed, y]
    # using data.load_data_to_bq()
    data_processed_with_id = pd.DataFrame(np.concatenate((
        data_clean[["ID"]],
        X_processed,
        y,
    ), axis=1))

    load_data_to_bq(
        data_processed_with_id,
        gcp_project=GCP_PROJECT,
        bq_dataset=BQ_DATASET,
        table=f'processed_{DATA_SIZE}',
        truncate=True
    )

    print("✅ preprocess() done \n")


def train(split_ratio: float = 0.02
    ) -> float:

    """
    - Download processed data from your BQ table (or from cache if it exists)
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return r2 (UNLESS WE DECIDE UPON A DIFFERENT MODEL)
    """

    # Load processed data using `get_data_with_cache`

    ###################### IMPORTANT #########################
    # THIS QUERY WILL NEED UPDATED TO FIT OUR ACCOUNT/DATA
    query = f"""
        SELECT * EXCEPT(_0)
        FROM {GCP_PROJECT}.{BQ_DATASET}.processed_{DATA_SIZE}
        ORDER BY _0 ASC
    """

    data_processed_cache_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed = get_data_with_cache(
        gcp_project=GCP_PROJECT,
        query=query,
        cache_path=data_processed_cache_path,
        data_has_header=False
    )

    if data_processed.shape[0] < 10:
        print("❌ Not enough processed data retrieved to train on")
        return None

    # Create (X_train_processed, y_train, X_val_processed, y_val)
    X_train, X_test, y_train, y_test = train_test_split(data_processed,
                                                        test_size=split_ratio,
                                                        random_state=42)

    # Train model using `model.py`
    model = load_model()

    if model is None:
        model = initialize_model()

    model = train_model(
        model, X_train, y_train
    )

    y_pred = model.predict(X_test)

    accuracy = round(accuracy_score(y_test, y_pred), 2)

    params = dict(
        context="train",
        training_shape=X_train.shape,
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=params, metrics=dict(accuracy=accuracy))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)

    print("✅ train() done \n")

    return accuracy

def pred(df: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if df is None:
        return "Unable to make prediction"

    model = load_model()
    assert model is not None

    X_processed = preprocess_features(df)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred
