import pandas as pd

'''
This module is used for cleaning data that can be
be used to pass into the model in the model.py function
'''

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean data by:
    - Removing or imputing
    '''
