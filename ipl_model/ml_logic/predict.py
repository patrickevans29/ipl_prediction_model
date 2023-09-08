from ipl_model.ml_logic.registry import load_model
import numpy as np
import pandas as pd
from ipl_model.ml_logic.preprocessor import preprocess_features, create_sklearn_preprocessor
import pickle

def pred(df: pd.DataFrame = None) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if df is None:
        return "Unable to make prediction"

    model = load_model()
    # assert model is not None
    # Load the saved preprocessor using pickle
    with open('preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)

    X_processed = preprocessor.transform(df)
    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred
