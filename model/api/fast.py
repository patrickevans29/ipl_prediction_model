import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model.ml_logic.model import pred

app = FastAPI()

# Allowing all middleware for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headersdir
)

@app.get("/predict")
def predict(
    # Here we will need to include all of the features
    # that we need to make a prediction
    team1: str,
    team2: str,
    team1_batting_average: float,
    team2_batting_average: float,
    city: str,
    season: str,
    venue: str,
    match_number: str,
    tosswinner: str,
    tossdecision: str
    ): # CHANGE OR ADD MORE IF NECESSARY
    '''
    Makes a prediction as to whether Team1 will win or not
    '''
    # Put the data into a dictionary to then put into a dataframe
    data = {
        "team1": [team1],
        "team2": [team2],
        "team1_batting_average": [team1_batting_average],
        "team2_batting_average": [team2_batting_average],
        "city": [city],
        "season": [season],
        "venue": [venue],
        "match_number": [match_number],
        "tosswinner": [tosswinner],
        "tossdecision": [tossdecision]
    }

    df = pd.DataFrame(data)

    prediction = pred(df)

    my_dict_return = {'Prediction': prediction}

    return my_dict_return
