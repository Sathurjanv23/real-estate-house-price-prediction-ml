import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    df = df.dropna()

    X = df.drop("price", axis=1)
    y = df["price"]

    X = pd.get_dummies(X, columns=["location"], drop_first=True)

    return X, y


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)