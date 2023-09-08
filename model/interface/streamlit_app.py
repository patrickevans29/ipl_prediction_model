import streamlit as st
import pandas as pd
import requests

'''
# IPLPredictionModel front

This front queries the Le Wagon IPL Prediction Model Team's model
(https://....ADD THE API HERE....)
'''

@st.cache_data # This will tell streamlit to keep the model in cache
def load_model():
    '''
    Load your trained model from ml-logic
    '''
    pass # ADD CODE HERE

model = load_model()
team_names = ['rajasthan royals', 'royal challengers bangalore',
       'sunrisers hyderabad', 'delhi capitals', 'chennai super kings',
       'gujarat titans', 'lucknow super giants', 'kolkata knight riders',
       'punjab kings', 'mumbai indians', 'rising pune supergiant',
       'gujarat lions', 'pune warriors', 'deccan chargers',
       'kochi tuskers kerala']

venues = ['narendra modi stadium, ahmedabad', 'eden gardens',
       'wankhede stadium', 'brabourne stadium',
       'dr dy patil sports academy',
       'maharashtra cricket association stadium',
       'dubai international cricket stadium', 'sharjah cricket stadium',
       'sheikh zayed stadium', 'arun jaitley stadium',
       'ma chidambaram stadium', 'rajiv gandhi international stadium',
       'dr. y.s. rajasekhara reddy aca-vdca cricket stadium',
       'punjab cricket association is bindra stadium',
       'm.chinnaswamy stadium', 'sawai mansingh stadium',
       'holkar cricket stadium', 'feroz shah kotla', 'green park',
       'saurashtra cricket association stadium',
       'shaheed veer narayan singh international stadium',
       'jsca international stadium complex',
       'sardar patel stadium, motera', 'barabati stadium',
       'subrata roy sahara stadium',
       'himachal pradesh cricket association stadium', 'nehru stadium',
       'vidarbha cricket association stadium, jamtha',
       'new wanderers stadium', 'supersport park', 'kingsmead',
       'outsurance oval', "st george's park", 'de beers diamond oval',
       'buffalo park', 'newlands']

# Create a simple Streamlit web app
st.title('IPL Match Prediction')

# Add user input elements
st.header('Choose Teams and Statistics')
st.subheader('Home Team Stats')
home_team = st.selectbox('Select Home Team', team_names)
home_batting_average = st.number_input('Select Home Team Batting Average')
home_toss_winner = st.radio("Did team1 win the toss?", [0,1])
home_points_against_avg = st.number_input('Select Average Runs Against Home Team')
home_mvp_average = st.number_input('Select Average Times Home Team has MVP')

st.subheader('Away Team Stats')
away_team = st.selectbox('Select Away Team', team_names)
away_batting_average = st.number_input('Select Away Team Batting Average')
away_points_against_avg = st.number_input('Select Average Runs Against Away Team')
away_mvp_average = st.number_input('Select Average Times Away Team has MVP')

st.subheader('Match Conditions')
venue = st.selectbox('Venue', venues)
match_importance = st.radio("How important is the match?", [1, 2, 3])
toss_decision = st.selectbox("What was the Toss Decision?", ["bat", "bowl"])

# Create a pandas DataFrame with user input
user_input_data = {
    'Team1': [home_team],
    'Team2': [away_team],
    'Team1_Batting_Average': [home_batting_average],
    'Team2_Batting_Average': [away_batting_average],
    'Team1_Toss_Winner': [home_toss_winner],
    'Team1_Points_Against_Avg': [home_points_against_avg],
    'Team2_Points_Against_Avg': [away_points_against_avg],
    'Team1_MVP_Average': [home_mvp_average],
    'Team2_MVP_Average': [away_mvp_average],
    'Venue': [venue],
    'MatchImportance': [match_importance],
    'TossDecision': [toss_decision]
}

user_input_df = pd.DataFrame(user_input_data)

# Write some code to get the other parameters needed for the model

params = None # ADD CODE

ipl_prediction_model_url = None # ADD CODE
response = requests.get(ipl_prediction_model_url, params=params)
prediction = response.json()

# Call the make the prediction
if st.button('Predict Winner'):
    st.write(f'Predicted Winner: {prediction}')
