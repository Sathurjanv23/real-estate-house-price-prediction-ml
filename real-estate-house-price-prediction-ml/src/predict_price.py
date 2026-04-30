import pickle
import pandas as pd

MODEL_PATH = "models/house_price_model.pkl"


def predict_house_price(area, bedrooms, bathrooms, location, house_age, parking):
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "house_age": [house_age],
        "parking": [parking],
        "location_Galle": [1 if location == "Galle" else 0],
        "location_Kandy": [1 if location == "Kandy" else 0],
        "location_Negombo": [1 if location == "Negombo" else 0],
    })

    prediction = model.predict(input_data)

    return prediction[0]


if __name__ == "__main__":
    price = predict_house_price(
        area=2200,
        bedrooms=4,
        bathrooms=3,
        location="Colombo",
        house_age=3,
        parking=2
    )

    print(f"Predicted House Price: {price:.2f}")