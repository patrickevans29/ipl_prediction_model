import numpy as np
import xgboost as xgb

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ipl_model.ml_logic.registry import *
from ipl_model.ml_logic.data import *
from ipl_model.params import *
from ipl_model.ml_logic.preprocessor import *
from ipl_model.ml_logic.feature_engineer import *

'''
Here is where we will initialize and train the model
'''

def initialize_model():
    '''
    Add our baseline or our final model with the correct parameters here
    '''
    grid = {'max_depth': 5,
       'base_score':0.5,
       'booster': 'gblinear',
        'gamma' : 0.1,
        'n_estimators': 150,
        'learning_rate': 0.1,
        'reg_lambda': 0.3,
       }

    model = xgb.XGBClassifier(params=grid)
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

# Engineer the team features that we need
engineered_df = feature_engineer(cleaned_df)

# Rename the columns
df = engineered_df.rename({'TeamA_batting_average': 'Team1_batting_average',
           'TeamB_batting_average': 'Team2_batting_average',
           'TeamA_innings_total': 'Team1_innings_total',
           'TeamB_innings_total' : 'Team2_innings_total'
          }, axis=1)

# WinningTeam = 1 --> Team1 Won
def map_winning_team(row):
    if row['WinningTeam'] == row['Team1']:
        return 1
    elif row['WinningTeam'] == row['Team2']:
        return 0
    else:
        return -1

df['WinningTeam'] = df.apply(map_winning_team, axis=1)
# drop rows with Winning Team = -1
df = df.drop(df[df['WinningTeam'] == -1].index)

# Toss Winner --> 0 = away team ; 1 = home team
def map_toss_winner(row):
    if row['TossWinner'] == row['Team1']:
        return 1
    elif row['TossWinner'] == row['Team2']:
        return 0
    else:
        return -1

df['TossWinner'] = df.apply(map_toss_winner, axis=1)

# Select only the columns that we need
df_to_model = df[COLUMN_NAMES]

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
