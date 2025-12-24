import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df.drop(columns=['cc_num', 'Unnamed: 0', 'trans_num'], inplace=True)

    return df


def check_class_imbalance(df, target_col='is_fraud'):
    class_counts = df[target_col].value_counts()
    fraud_percentage = class_counts[1] / class_counts.sum() * 100
    return class_counts, fraud_percentage


def train_test_data_split(X, y, test_size=0.2):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )


def reduce_high_cardinality(X_train, X_test, max_categories=100):
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        if X_train[col].nunique() > max_categories:
            top_categories = X_train[col].value_counts().head(max_categories).index
            X_train[col] = X_train[col].apply(lambda x: x if x in top_categories else 'Other')
            X_test[col] = X_test[col].apply(lambda x: x if x in top_categories else 'Other')

    return X_train, X_test
