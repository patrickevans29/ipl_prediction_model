import streamlit as st
import pandas as pd
import requests

info = '''
# IPLPredictionModel front

This front queries the Le Wagon IPL Prediction Model Team's model
(https://....ADD THE API HERE....)
'''

# Set page title and favicon
st.set_page_config(
    page_title="IPL Match Prediction",
    page_icon="🏏",
)

team_logos = {
    'gujarat titans': 'streamlit_app/logos/Chennai_Super_Kings_Logo.svg.png',
    'rajasthan royals': 'https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg',
    'lucknow super giants': 'https://upload.wikimedia.org/wikipedia/en/thumb/a/a9/Lucknow_Super_Giants_IPL_Logo.svg/1200px-Lucknow_Super_Giants_IPL_Logo.svg.png',
    'punjab kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/d/d4/Punjab_Kings_Logo.svg/1200px-Punjab_Kings_Logo.svg.png',
    'mumbai indians': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/cd/Mumbai_Indians_Logo.svg/1200px-Mumbai_Indians_Logo.svg.png',
    'royal challengers bangalore': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2a/Royal_Challengers_Bangalore_2020.svg/1200px-Royal_Challengers_Bangalore_2020.svg.png',
    'kolkata knight riders': 'https://upload.wikimedia.org/wikipedia/en/thumb/4/4c/Kolkata_Knight_Riders_Logo.svg/1200px-Kolkata_Knight_Riders_Logo.svg.png',
    'sunrisers hyderabad': 'https://upload.wikimedia.org/wikipedia/en/e/eb/Sunrisers_Hyderabad.png',
    'delhi capitals': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2f/Delhi_Capitals.svg/1200px-Delhi_Capitals.svg.png',
    'chennai super kings': 'https://upload.wikimedia.org/wikipedia/en/thumb/2/2b/Chennai_Super_Kings_Logo.svg/640px-Chennai_Super_Kings_Logo.svg.png',
    'rising pune supergiant': 'https://upload.wikimedia.org/wikipedia/en/9/9a/Rising_Pune_Supergiant.png',
    'gujarat lions': 'https://upload.wikimedia.org/wikipedia/en/thumb/c/c4/Gujarat_Lions.png/200px-Gujarat_Lions.png',
    'pune warriors': 'https://upload.wikimedia.org/wikipedia/en/4/4a/Pune_Warriors_India_IPL_Logo.png',
    'deccan chargers': 'https://upload.wikimedia.org/wikipedia/en/a/a6/HyderabadDeccanChargers.png',
    'kochi tuskers kerala': 'https://upload.wikimedia.org/wikipedia/en/thumb/9/96/Kochi_Tuskers_Kerala_Logo.svg/1200px-Kochi_Tuskers_Kerala_Logo.svg.png'
}

# Team names for selection
team_names = ['rajasthan royals', 'royal challengers bangalore',
       'sunrisers hyderabad', 'delhi capitals', 'chennai super kings',
       'gujarat titans', 'lucknow super giants', 'kolkata knight riders',
       'punjab kings', 'mumbai indians', 'rising pune supergiant',
       'gujarat lions', 'pune warriors', 'deccan chargers',
       'kochi tuskers kerala']

# Capitalized version of the team names for display
team_names_display = [team_name.title() for team_name in team_names]

# Dictionary to map capitalized names to uncapitalized names
team_names_mapping = {display_name: name for display_name, name in zip(team_names_display, team_names)}
# Dictionary to map capitalized names to uncapitalized names
team_names_capitalise = {name: display_name for name, display_name in zip(team_names, team_names_display)}

# Possible venues
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

# Capitalized version of the venues names for display
venue_names_display = [venue_name.title() for venue_name in venues]

# Dictionary to map capitalized names to uncapitalized names
venue_names_mapping = {display_name: name for display_name, name in zip(venue_names_display, venues)}
# Dictionary to map capitalized names to uncapitalized names
team_names_capitalise = {name: display_name for name, display_name in zip(venues, venue_names_display)}

cities = ['ahmedabad', 'kolkata', 'mumbai', 'navi mumbai', 'pune', 'dubai',
       'sharjah', 'abu dhabi', 'delhi', 'chennai', 'hyderabad',
       'visakhapatnam', 'chandigarh', 'bengaluru', 'jaipur', 'indore',
       'bangalore', 'kanpur', 'rajkot', 'raipur', 'ranchi', 'cuttack',
       'dharamsala', 'kochi', 'nagpur', 'johannesburg', 'centurion',
       'durban', 'bloemfontein', 'port elizabeth', 'kimberley',
       'east london', 'cape town']

# Capitalized version of the venues names for display
city_names_display = [city_name.title() for city_name in cities]

# Dictionary to map capitalized names to uncapitalized names
city_names_mapping = {display_name: name for display_name, name in zip(city_names_display, cities)}
# Dictionary to map capitalized names to uncapitalized names
city_names_capitalise = {name: display_name for name, display_name in zip(cities, city_names_display)}

# Create a simple Streamlit web app
st.title('IPL Match Prediction')

# Add user input elements
st.header('Choose Teams and Statistics')

# Select columns
col1, col2, col3 = st.columns(3)

# Adjust column 2 style to make text bigger and in bold
col2.markdown('<style>div[data-testid="stHorizontalBlock"] > div{display: flex; flex-direction: column; justify-content: center; text-align: center;}</style>', unsafe_allow_html=True)
col2.markdown('<style>div[data-testid="stHorizontalBlock"] > div p{font-size: 20px; font-weight: bold;}</style>', unsafe_allow_html=True)

with col1:
    selected_team_1_display = st.selectbox('Select Team 1', team_names_display)
    team_1 = team_names_mapping[selected_team_1_display]
    st.image(team_logos[team_1], width=100)

with col2:
    st.write("VS.")

with col3:
    selected_team_2_display = st.selectbox('Select Team 2', team_names_display)
    team_2 = team_names_mapping[selected_team_2_display]
    st.image(team_logos[team_2], width=100)

st.subheader(f'{team_1} Stats')
team_1_score = st.number_input(f'Select {team_1} Weighted Score')
team_1_batting_average = st.number_input(f'Select {team_1} Weighted Batting Average')
team_1_batting_strike = st.number_input(f'Select {team_1} Weighted Strike Rate')
team_1_bowling_average = st.number_input(f'Select {team_1} Bowling Average')
team_1_bowling_economy = st.number_input(f'Select {team_1} Bowling Economy')
team_1_win_ratio = st.number_input(f'Select {team_1} Win Ratio')

st.subheader(f'{team_2} Stats')
team_2_score = st.number_input(f'Select {team_2} Weighted Score')
team_2_batting_average = st.number_input(f'Select {team_2} Weighted Batting Average')
team_2_batting_strike = st.number_input(f'Select {team_2} Weighted Strike Rate')
team_2_bowling_average = st.number_input(f'Select {team_2} Bowling Average')
team_2_bowling_economy = st.number_input(f'Select {team_2} Bowling Economy')
team_2_win_ratio = st.number_input(f'Select {team_2} Win Ratio')

st.subheader('Select Match Conditions')

# Create two columns to improve readability
col1, col2 = st.columns(2)
with col1:
    city_display_name = st.selectbox('City', city_names_display)
    city = city_names_mapping[city_display_name]
    venue_display_name = st.selectbox('Venue', venue_names_display)
    venue = venue_names_mapping[venue_display_name]
with col2:
    toss_winner_display = st.selectbox("Which team won the toss?", [selected_team_1_display, selected_team_2_display])
    toss_winner = team_names_mapping[toss_winner_display]
    toss_decision = st.selectbox("What was the Toss Decision?", ["bat", "field"])

# Edit toss_winner so that input is 1 or 0
toss_winner = [1 if toss_winner==team_1 else 0]

# Create a pandas DataFrame with user input
user_input_data = {
        'Team1': [team_1],
        'Team2': [team_2],
        'Avg_Weighted_Score_diff': [team_1_score-team_2_score],
        'batting_average_weighted_diff': [team_1_batting_average-team_2_batting_average],
        'batting_strike_rate_weighted_diff': [team_1_batting_strike-team_2_batting_strike],
        'bowling_average_diff': [team_1_bowling_average-team_2_bowling_average],
        'bowling_economy_rate_diff': [team_1_bowling_economy-team_2_bowling_economy],
        'win_ratio_diff': [team_1_win_ratio-team_2_win_ratio],
        'Venue': [venue],
        'City': [city],
        'TossWinner': [toss_winner],
        'TossDecision': [toss_decision]
    }

user_input_df = pd.DataFrame(user_input_data)

# Write some code to get the other parameters needed for the model


ipl_prediction_model_url = "https://iplpredictionmodel-wvacvcx6pq-ew.a.run.app/predict" # Running locally
response = requests.get(ipl_prediction_model_url, params=user_input_data)
prediction = response.json()
prediction = [team_1 if prediction==1.0 else team_2]
prediction = prediction[0]

# Call the make the prediction
if st.button('Predict Winner'):
    st.header('Prediction Result')
    st.markdown('### The model predicts that the winner will be...')

    # Display the team logo of the predicted winner
    if prediction == team_1:
        predicted_team_logo = team_logos[team_1]
    elif prediction == team_2:
        predicted_team_logo = team_logos[team_2]
    else:
        predicted_team_logo = None

    # Display the prediction result with the team logo
    if predicted_team_logo:
        st.image(predicted_team_logo, width=150)
    prediction = [selected_team_1_display if prediction == team_1 else selected_team_2_display]
    st.write(prediction[0])


nothing = """
### FOR REFERENCE PURPOSES
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
"""
