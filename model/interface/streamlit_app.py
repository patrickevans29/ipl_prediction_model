import streamlit as st
import pandas as pd
import numpy as np

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
st.header('Choose Teams and Venue')
home_team = st.selectbox('Select Home Team', team_names)
away_team = st.selectbox('Select Away Team', team_names)
venue = st.selectbox('Select Venue', venues)

# Save the user input to put into the model
user_input = pd.DataFrame({'team1': [home_team], 'team2': [away_team], 'venue': [venue]})

# Call the make the prediction
if st.button('Predict Winner'):
    prediction = model.predict(user_input)
    st.write(f'Predicted Winner: {prediction}')
