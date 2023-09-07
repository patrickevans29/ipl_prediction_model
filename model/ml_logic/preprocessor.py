import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    '''
    Depends on the model that we choose.
    This should transform the cleaned dataset by normalising the data.
    The output should have many more features due to OneHotEncoding.
    '''
    def create_sklearn_preprocessor() -> ColumnTransformer:
        '''
        To be able to process the data we can use an SKlearn pipeline
        '''
        pass # ADD CODE HERE should return something like final_preprocessor

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed
