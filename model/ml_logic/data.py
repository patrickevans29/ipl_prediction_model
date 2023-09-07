import pandas as pd
import numpy as np

'''
This module is used for cleaning data that can be
be used to pass into the model in the model.py function
'''

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean data by:
    - Removing or imputing NaN
    '''
    ## cleaning data from ball_by_ball dataframe
    # drop duplicates
    df = df.drop_duplicates()
    # assign the good value to overs
    df["overs"] = df["overs"] + 1
    # cleaning the string values
    # batter
    df["batter"] = [text.strip() for text in df["batter"]] # removing space
    df["batter"] = [text.lower() for text in df["batter"]] # lowercase 
    # bowler
    df["bowler"] = [text.strip() for text in df["bowler"]]
    df["bowler"] = [text.lower() for text in df["bowler"]]
    # non-striker
    df.rename(columns={"non-striker": "non_striker"}, inplace=True) #renaming the non_striker
    df["non_striker"] = [text.strip() for text in df["non_striker"]]
    df["non_striker"] = [text.lower() for text in df["non_striker"]]
    # extra-types
    df["extra_type"] = df["extra_type"].fillna("noextra") # filling the NaN values
    # player out
    df["player_out"] = df["player_out"].fillna("noplayerout") # filling the NaN values
    df["player_out"] = [text.strip() for text in df["player_out"]]
    df["player_out"] = [text.lower() for text in df["player_out"]]
    # kind
    df["kind"] = df["kind"].fillna("nokind") # filling the NaN values
    df["kind"] = [text.strip() for text in df["kind"]]
    df["kind"] = [text.lower() for text in df["kind"]]
    # fielders involved
    df["fielders_involved"] = df["fielders_involved"].fillna("nofieldersinvolved") ## filling the NaN values
    df["fielders_involved"] = [text.strip() for text in df["fielders_involved"]]
    df["fielders_involved"] = [text.lower() for text in df["fielders_involved"]]
    # batting team
    df["BattingTeam"] = [text.strip() for text in df["BattingTeam"]]
    df["BattingTeam"] = [text.lower() for text in df["BattingTeam"]]
    # some teams change their name since 2008
    team_name_dict = {'rising pune supergiants': 'rising pune supergiant', 
                  'kings xi punjab': 'punjab kings',
                  'delhi daredevils': 'delhi capitals'}
    df['BattingTeam'] = df['BattingTeam'].replace(team_name_dict)
    
    ## cleaning data from match dataframe
    # Adding the correct City for the matching Venue
    df.loc[df.Venue == 'Dubai International Cricket Stadium','City'] = 'Dubai'
    df.loc[df.Venue == 'Sharjah Cricket Stadium','City'] = 'Sharjah'
    # Creating a dictionary of duplicated Venues so only one entry is created for each venue
    venues_dict = {'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
                'Brabourne Stadium, Mumbai': 'Brabourne Stadium',
                'Dr DY Patil Sports Academy, Mumbai': 'Dr DY Patil Sports Academy',
                'Eden Gardens, Kolkata': 'Eden Gardens',  
                'M Chinnaswamy Stadium': 'M.Chinnaswamy Stadium',
                'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
                'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium',
                'Maharashtra Cricket Association Stadium, Pune': 'Maharashtra Cricket Association Stadium',
                'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
                'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association IS Bindra Stadium',
                'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
                'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
                'Zayed Cricket Stadium, Abu Dhabi': 'Sheikh Zayed Stadium'}
    df['Venue'] = df['Venue'].replace(venues_dict)
    # Two teams have changed their name over the period the dataset covers
    team_name_changes = {'Kings XI Punjab': 'Punjab Kings',
                        'Delhi Daredevils': 'Delhi Capitals'}
    df['Team1'] = df['Team1'].replace(team_name_changes)
    # Updating the team name for Rising Pune Supergiant
    team_name_dict = {'Rising Pune Supergiants': 'Rising Pune Supergiant'}
    df['Team1'] = df['Team1'].replace(team_name_dict)
    # Updating the other team name columns
    df['Team2'] = df['Team2'].replace(team_name_changes)
    df['TossWinner'] = df['TossWinner'].replace(team_name_changes)
    df['WinningTeam'] = df['WinningTeam'].replace(team_name_changes)
    df['Team2'] = df['Team2'].replace(team_name_dict)
    df['TossWinner'] = df['TossWinner'].replace(team_name_dict)
    df['WinningTeam'] = df['WinningTeam'].replace(team_name_dict)
    # Identifying and removing 4 "NoResult" Values
    df['WinningTeam'] = df['WinningTeam'].fillna('NoResults')
    df['Player_of_Match'] = df['Player_of_Match'].fillna('NoResults')
    df['SuperOver'] = df['SuperOver'].fillna('NoResults')
    df.loc[df['WinningTeam'].isna(), 'method'] = 'NoResults'
    df.loc[df['WinningTeam'].isna(), 'Margin'] = 0
    # The method column held no useful information and was dropped
    df = df.drop(columns='method')
    # Results from margin related to Superover and have been filled with 0
    df['Margin'] = df['Margin'].fillna(0)
    
    # Updating dates so we are able to identify seasonal changes in data
    season_dict = {'2020/21': '2020',
                '2009/10': '2010',
                '2007/08': '2008'}

    df['Season'] = df['Season'].replace(season_dict)
    # Selecting columns to strip
    df_columns = list(df.select_dtypes(include='object').columns)
    # Stripping the data for each column
    for column in df_columns:
        df[column] = [text.strip().lower() for text in df[column]]
    
    print("✅ data cleaned")
    return df
    

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This is where we create the new features for the model.
    They should be:
    - team_1_win --> 1 or 0 if team 1 one the match
    - innings_total --> to be used to calculate batting avgs etc.
    - team_1_batting_average --> the average amount of runs that team1 scores in a match
    - team_2_batting_average --> the average amount of runs that team2 scores in a match
    - team_1_toss_winner --> 1 or 0 if team1 won the toss
    - match_importance --> ordinal value given to the importance of the match. Final or regular etc.
    - team_1_points_against_avg --> the average number of point opposition teams score agains team_1
    - team_2_points_against_avg --> the average number of point opposition teams score agains team_2
    - team_1_avg_mvp --> average number of times that team_1 has MVP
    - team_2_avg_mvp --> average number of times that team_2 has MVP
    '''
    # create the binary column Team1_Win
    df['Team1_Win'] = (df['Team1'] == df['WinningTeam']).astype(int)
    # create the innings total
    df['innings_total'] = df.groupby(['ID', 'innings'])['total_run'].transform('sum')
    df = df[(df['innings'] == 1) | (df['innings'] == 2)]
    # Convert the innings totals into columns

    # Group by 'ID' and 'innings' and calculate innings_total
    innings_totals = df.groupby(['ID', 'innings'])['innings_total'].mean().unstack()
    innings_totals.columns = ['Team1_innings_total', 'Team2_innings_total']

    # Merge the innings_totals DataFrame back into new_df on 'ID'
    df = pd.merge(df, innings_totals, left_on='ID', right_index=True)
    df['team_batting_average'] = df.groupby('BattingTeam')['total_run'].transform('sum') / df.groupby('BattingTeam')['ID'].transform('nunique')
    # Extract the relevant columns 
    selected_columns = ['ID', 'innings', 'Team1', 'Team2', 'team_batting_average', 'MatchNumber','WinningTeam', 'innings_total', 'City', 'Date', 'Venue', 'TossWinner', 'TossDecision', 'Team1Players', 'Team2Players']
    new_df = df[selected_columns].copy()
    new_df.head()

    # Filter innings 1 (Team1's innings)
    team_a_data = new_df[new_df['innings'] == 1][['ID','Team1', 'team_batting_average' ]]
    team_a_data.rename(columns={'team_batting_average': 'Team_1_batting_average'}, inplace=True)

    # Filter innings 2 (Team1's innings)
    team_b_data = new_df[new_df['innings'] == 2][['ID', 'Team2', 'team_batting_average']]
    team_b_data.rename(columns={'team_batting_average': 'Team_2_batting_average'}, inplace=True)

    # Merge the data from innings 1 and innings 2 for each match ID
    merged_data = pd.merge(team_a_data, team_b_data, on='ID')
    # Drop duplicates based on 'ID' column and keep the first occurrence
    final_data = merged_data.drop_duplicates(subset='ID', keep='first')
    # Reset the index
    final_data.reset_index(drop=True, inplace=True)
    # drop the duplicate columns
    final_data.drop(columns=["Team1", "Team2"], inplace=True)
    # merge with the main df 
    df = pd.merge(df, final_data, on = "ID")
    # drop the useless column
    df.drop(columns=['team_batting_average'], inplace=True)
    # new binary column for the toss winner
    df['team_1_toss_winner'] = (df['Team1'] == df['TossWinner']).astype(int)
    
    # Apply the function
    df['MatchImportance'] = df['MatchNumber'].apply(map_match_number)
    # Calculate the average points scored against a each team
    df['Team1_points_against_avg'] = df.groupby('Team1')['Team2_innings_total'].transform('mean')
    df['Team2_points_against_avg'] = df.groupby('Team2')['Team1_innings_total'].transform('mean')
    # Calculate the average number of times that team_1 has MVP
    
    df['Team_MVP'] = df.apply(lambda row: get_match_winner(row['Player_of_Match'], row['Team1Players'], row['Team2Players']), axis=1)
    
    df['Team_MVP'] = df.apply(replace_team_mvp_with_name, axis=1)
    team_mvp_counts = df['Team_MVP'].value_counts().reset_index()

    team_mvp_counts.columns = ['Team_MVP', 'MVP_Count']

    # Create the columns Team1_MVP_appearances e Team2_MVP_appearances
    df = df.merge(team_mvp_counts, left_on='Team1', right_on='Team_MVP', how='left').fillna(0)
    df.rename(columns={'MVP_Count': 'Team1_MVP_appearances'}, inplace=True)

    df = df.merge(team_mvp_counts, left_on='Team2', right_on='Team_MVP', how='left').fillna(0)
    df.rename(columns={'MVP_Count': 'Team2_MVP_appearances'}, inplace=True)

    # Drop columns
    df.drop(['Team_MVP_x', 'Team_MVP_y', 'Team_MVP'], axis=1, inplace=True)

    # Now, let's create the average of MVP appearences
    total_games_team1 = df['Team1'].value_counts()
    total_games_team2 = df['Team2'].value_counts()

    total_games = total_games_team2 + total_games_team1

    df['Team1_MVP_average'] = df['Team1_MVP_appearances'] / total_games[df['Team1']].values
    df['Team2_MVP_average'] = df['Team2_MVP_appearances'] / total_games[df['Team2']].values
    df = df.drop(columns = ['Team1_MVP_appearances', 'Team2_MVP_appearances'])
    
    df = clean_data(df)
    return df

def get_data(userinput: list) -> pd.DataFrame:
    '''
    Here is where the data is taken. This could be based on the user input
    on Streamlit. The information could be taken from BigQuery once the user
    asks for a prediction.
    '''
    pass # ADD CODE HERE

# define the importance of each match 
def map_match_number(value):
            if isinstance(value, int) or value.isnumeric():
                return 1
            elif value in ['qualifier 2', 'eliminator', 'qualifier 1', 'qualifier', 'elimination final', '3rd place play-off', 'semi final']:
                return 2
            elif value == 'final':
                return 3
            else:
                return 0

def get_match_winner(player_of_match, team1_players, team2_players):
            if pd.isna(player_of_match):
                return 'N/A'  
            
            if pd.isna(team1_players):
                team1_players = []  
            
            if pd.isna(team2_players):
                team2_players = [] 
            
            if player_of_match in team1_players:
                return 'Team1'
            elif player_of_match in team2_players:
                return 'Team2'
            else:
                return 'N/A'
            
def replace_team_mvp_with_name(row):
            if row['Team_MVP'] == 'Team1':
                return row['Team1']
            elif row['Team_MVP'] == 'Team2':
                return row['Team2']
            else:
                return row['Team_MVP']