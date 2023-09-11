import numpy as np
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ipl_model.ml_logic.registry import *
from ipl_model.ml_logic.data import *
from ipl_model.params import *
from ipl_model.ml_logic.preprocessor import *

'''
Here is where we will initialize and train the model
'''

def initialize_model():
    '''
    Add our baseline or our final model with the correct parameters here
    '''
    model = XGBClassifier(colsample_bytre = 0.8,
                          learning_rate = 0.01,
                          max_depth = 4,
                          n_estimators = 100,
                          subsample = 0.8)
    print("✅ Model initialized")

    return model

def train_model(model,
        X: np.ndarray,
        y: np.ndarray):

    '''
    Train the model with this function. Training data should
    be cleaned and engineered and preprocessed.
    '''
    model = model.fit(X, y)

    return model

# Get the raw data
complete_df = load_data_locally()

# Clean the data
cleaned_df = clean_data(complete_df)

# Engineer the features that we need
clean_engineered_df = feature_engineer(cleaned_df)

# Select only the columns that we need
df_to_model = clean_engineered_df[COLUMN_NAMES]

# Create X and y
X = df_to_model[FEATURE_NAMES]
y = df_to_model[TARGET_NAME]

# Preprocess X
X_processed = preprocess_features(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# initialize the model
model = initialize_model()

# Train the model on the X_train and y_train
trained_model = train_model(model, X_train, y_train)

# Make predictions on the testing set
y_pred = trained_model.predict(X_test)

# Evaluate the model
accuracy = round(accuracy_score(y_test, y_pred), 2)

# Save the model locally
save_model(trained_model)

print(f"Model with accuracy score of {accuracy} saved to {LOCAL_MODELS_PATH}")
