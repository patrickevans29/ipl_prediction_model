import pandas as pd
from google.cloud import bigquery
from pathlib import Path

'''
This module is used for cleaning data that can be
be used to pass into the model in the model.py function
'''

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Clean data by:
    - Removing or imputing NaN
    '''
    pass # ADD CODE HERE

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
    df = clean_data(df)
    pass # ADD CODE HERE

def get_data_with_cache(
        gcp_project:str,
        query:str,
        cache_path:Path,
        data_has_header=True
    ) -> pd.DataFrame:

    """
    Retrieve `query` data from BigQuery, or from `cache_path` if the file exists
    Store at `cache_path` if retrieved from BigQuery for future use
    """

    if cache_path.is_file():
        print("\nLoad data from local CSV...")
        df = pd.read_csv(cache_path, header='infer' if data_has_header else None)
    else:
        print("\nLoad data from BigQuery server...")
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(cache_path, header=data_has_header, index=False)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        table: str,
        truncate: bool
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(f"\nSave data to BigQuery @ {full_table_name}...:")

    # Load data onto full_table_name
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")

'''
IGNORE THIS FOR NOW

def get_data(userinput: list) -> pd.DataFrame:

    Here is where the data is taken. This could be based on the user input
    on Streamlit. The information could be taken from BigQuery once the user
    asks for a prediction.

    pass # ADD CODE HERE
    '''
