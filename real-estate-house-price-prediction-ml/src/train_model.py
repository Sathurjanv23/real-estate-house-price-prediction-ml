import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from data_preprocessing import load_data, preprocess_data, split_data


DATA_PATH = "data/house_data.csv"
MODEL_PATH = "models/house_price_model.pkl"


def train_model():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Model Training Completed ✅")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    os.makedirs("models", exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()