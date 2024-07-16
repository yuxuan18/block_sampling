import pandas as pd

def clean_none(df: pd.DataFrame, column: str) -> pd.DataFrame:
    n_bad_data = df[column].isna().sum()
    print(f"Found {n_bad_data} bad data in {column}")
    df[column] = df[column].apply(lambda x: str(x) if len(str(x)) > 0 else "None")
    return df