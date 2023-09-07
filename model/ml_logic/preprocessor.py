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
    ## defining the list of categorical and numerical features 
    categorical_columns = ['City', 'MatchNumber', 'Season', 'Team1', 'Team2', 'Venue', 'TossWinner', 'TossDecision']
    numerical_columns = ['TeamA_batting_average','TeamB_batting_average']
    ## apply the encoder, OneHotEncoder for categorical and the RobustScaler for numerical features
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
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
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape", X_processed.shape)
    return X_processed
