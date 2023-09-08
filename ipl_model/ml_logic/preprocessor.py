import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from ipl_model.params import *

def preprocess_features(X):
    '''
    Depends on the model that we choose.
    This should transform the cleaned dataset by normalizing the data.
    The output should have many more features due to OneHotEncoding.
    '''
    ## defining the list of categorical and numerical features
    categorical_columns = CATAGORICAL_COLUMNS
    numerical_columns = NUMERICAL_COLUMNS

    ## apply the encoder, OneHotEncoder for categorical and the RobustScaler for numerical features
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline([
        ('scaler', RobustScaler())
    ])

    def create_sklearn_preprocessor() -> ColumnTransformer:
        '''
        To be able to process the data we can use an SKlearn pipeline
        '''
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_columns),
                ('num', numerical_transformer, numerical_columns)
            ])

        ## create the pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        return pipeline

    preprocessor = create_sklearn_preprocessor()

    if isinstance(X, pd.DataFrame):
        data_processed = preprocessor.fit_transform(X)
    elif isinstance(X, np.ndarray):
        data_processed = preprocessor.fit_transform(pd.DataFrame(X, columns=numerical_columns + categorical_columns))
    else:
        raise ValueError("Unsupported input type. X must be a pandas DataFrame or a numpy array.")

    print("✅ X_processed, with shape", data_processed.shape)
    return data_processed
