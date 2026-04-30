import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_data, preprocess_data, split_data


DATA_PATH = "data/house_data.csv"
MODEL_PATH = "models/house_price_model.pkl"
OUTPUT_PATH = "outputs/prediction_results.csv"


def evaluate_model():
    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(X_test)

    results = pd.DataFrame({
        "Actual Price": y_test.values,
        "Predicted Price": predictions
    })

    results.to_csv(OUTPUT_PATH, index=False)

    print("Model Evaluation ✅")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")
    print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate_model()