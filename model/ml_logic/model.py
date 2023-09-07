import numpy as np
import xgboost as xgb
import pandas as pd
from xgboost import XGBClassifier


'''
Here is where we will initialize and train the model
'''

def initialize_model() -> xgb.Model:
    '''
    Add our baseline or our final model with the correct parameters here
    '''
    model = XGBClassifier(colsample_bytre = 0.8,
                          learning_rate = 0.01,
                          max_depth = 4,
                          n_estimators = 100,
                          subsample = 0.8)
    print("✅ Model initialized")

def train_model(model: Model,
        X: np.ndarray,
        y: np.ndarray):

    '''
    Train the model with this function. Training data should
    be cleaned and engineered and preprocessed.
    '''
    model = model.fit(X, y)
    
    return model
