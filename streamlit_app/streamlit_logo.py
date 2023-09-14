# team logos

import streamlit as st

team_logos = {
    # ... (your team logos)
}

# Function to display the logo and allow selection
def select_team_with_logo(team_names, team_logos):
    st.subheader('Select a Team')
    selected_team = st.selectbox('Select a team:', team_names, format_func=lambda team: st.image(team_logos[team], width=150), key='team_select')
    return selected_team

# List of team names
team_names = list(team_logos.keys())

# Use the function to allow the user to select a team
selected_team = select_team_with_logo(team_names, team_logos)

st.write(f'Selected Team: {selected_team}')

"""
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
