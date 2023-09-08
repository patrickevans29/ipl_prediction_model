import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ipl_model.params import *

def create_sklearn_preprocessor() -> ColumnTransformer:
    '''
    To be able to process the data we can use an SKlearn pipeline
    '''
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline([
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, CATAGORICAL_COLUMNS),
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

    # Save the transformer for later
    with open('preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    return data_processed
