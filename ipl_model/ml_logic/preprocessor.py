import numpy as np
import pandas as pd
import pickle
import time

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
from google.cloud import storage

from ipl_model.params import *

def create_sklearn_preprocessor() -> ColumnTransformer:
    '''
    Create a pipeline to process the data
    '''
    categorical_transformer1 = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    '''
    categorical_transformer2 = Pipeline([
        ('label_encode', LabelEncoder())
    ])
    '''

    numerical_transformer = Pipeline([
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat1', categorical_transformer1, CATAGORICAL_COLUMNS),
            ('num', numerical_transformer, NUMERICAL_COLUMNS)
        ])

    ## create the pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    return pipeline

# Use this function to preprocess training data
def preprocess_features(X):
    preprocessor = create_sklearn_preprocessor()
    if isinstance(X, pd.DataFrame):
        data_processed = preprocessor.fit_transform(X)
    elif isinstance(X, np.ndarray):
        data_processed = preprocessor.fit_transform(pd.DataFrame(X, columns=NUMERICAL_COLUMNS + CATAGORICAL_COLUMNS))
    else:
        raise ValueError("Unsupported input type. X must be a pandas DataFrame or a numpy array.")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    preprocessor_path = os.path.join(LOCAL_PREPROCESSORS_PATH, f"{timestamp}_preprocessor.pkl")

    # Create the directory if it doesn't exist
    os.makedirs(LOCAL_PREPROCESSORS_PATH, exist_ok=True)

    # Save the preprocessor locally using pickle
    with open(preprocessor_path, 'wb') as preprocessor_file:
        pickle.dump(preprocessor, preprocessor_file)

    print("✅ Preprocessor saved locally")

    # Save the preprocessor to GCS
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"preprocessors/{timestamp}_preprocessor.pkl")
    blob.upload_from_filename(preprocessor_path)

    print("✅ Preprocessor saved to GCS")

    return data_processed
