import pandas as pd

def engineer_features(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    # Time-based features
    df['transaction_hour'] = df['trans_date_trans_time'].dt.hour
    df['transaction_day'] = df['trans_date_trans_time'].dt.day
    df['transaction_month'] = df['trans_date_trans_time'].dt.month

    # Age feature
    df['customer_age'] = (
        df['trans_date_trans_time'] - df['dob']
    ).dt.days // 365

    # Drop original datetime columns
    df.drop(['trans_date_trans_time', 'dob'], axis=1, inplace=True)

    return df
