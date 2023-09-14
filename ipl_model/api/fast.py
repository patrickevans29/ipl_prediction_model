import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ipl_model.ml_logic.predict import pred

app = FastAPI()

# Allowing all middleware for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headersdir
)

@app.get("/")
def hello():
    return {"Hello": "You"}

@app.get("/predict")
def predict(
    Team1: str,
    Team2: str,
    Avg_Weighted_Score_diff: float,
    batting_average_weighted_diff: float,
    batting_strike_rate_weighted_diff,
    bowling_average_diff: float,
    bowling_economy_rate_diff: float,
    win_ratio_diff: float,
    Venue: str,
    City: str,
    TossWinner: float,
    TossDecision: str):

    '''
    Makes a prediction as to whether Team1 will win or not using the pred()
    function from main.py
    '''

    user_input_data = {
        'Team1': [Team1],
        'Team2': [Team2],
        'Avg_Weighted_Score_diff': [Avg_Weighted_Score_diff],
        'batting_average_weighted_diff': [batting_average_weighted_diff],
        'batting_strike_rate_weighted_diff': [batting_strike_rate_weighted_diff],
        'bowling_average_diff': [bowling_average_diff],
        'bowling_economy_rate_diff': [bowling_economy_rate_diff],
        'win_ratio_diff': [win_ratio_diff],
        'Venue': [Venue],
        'City': [City],
        'TossWinner': [TossWinner],
        'TossDecision': [TossDecision]
    }

    df = pd.DataFrame(user_input_data)

    # Ensure the data has the correct shape (1, 12) for prediction
    if df.shape != (1, 12):
        raise HTTPException(status_code=400, detail="Invalid data shape")

    prediction = float(pred(df))

    my_dict_return = {'Prediction': prediction}

    return my_dict_return
