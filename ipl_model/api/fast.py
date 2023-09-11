import pandas as pd
import numpy as np

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
    Team_1_batting_average: float,
    Team_2_batting_average: float,
    team_1_toss_winner: float,
    Team1_points_against_avg: float,
    Team2_points_against_avg: float,
    Team1_MVP_average: float,
    Team2_MVP_average: float,
    Venue: str,
    MatchImportance: float,
    TossDecision: str
):
    '''
    Makes a prediction as to whether Team1 will win or not using the pred()
    function from main.py
    '''
    user_input_data = {
        'Team1': [Team1],
        'Team2': [Team2],
        'Team_1_batting_average': [Team_1_batting_average],
        'Team_2_batting_average': [Team_2_batting_average],
        'team_1_toss_winner': [team_1_toss_winner],
        'Team1_points_against_avg': [Team1_points_against_avg],
        'Team2_points_against_avg': [Team2_points_against_avg],
        'Team1_MVP_average': [Team1_MVP_average],
        'Team2_MVP_average': [Team2_MVP_average],
        'Venue': [Venue],
        'MatchImportance': [MatchImportance],
        'TossDecision': [TossDecision]
    }

    df = pd.DataFrame(user_input_data)

    # Ensure the data has the correct shape (1, 12) for prediction
    if df.shape != (1, 12):
        raise HTTPException(status_code=400, detail="Invalid data shape")

    prediction = float(pred(df))

    my_dict_return = {'Prediction': prediction}

    return my_dict_return
