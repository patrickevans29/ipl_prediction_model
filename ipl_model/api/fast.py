import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ipl_model.interface.main import pred

app = FastAPI()

# Set a long timeout
app.add_middleware(TimeoutMiddleware, timeout_seconds=60)

# Allowing all middleware for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headersdir
)

@app.get("/predict")
def predict(user_input_data: dict)->dict:
    '''
    Makes a prediction as to whether Team1 will win or not using the pred()
    function from main.py
    '''
    df = pd.DataFrame(user_input_data)

    prediction = pred(df)

    my_dict_return = {'Prediction': prediction}

    return my_dict_return
