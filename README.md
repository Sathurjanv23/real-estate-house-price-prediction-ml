# 🏠 Real Estate House Price Prediction ML

This is an internship-level Machine Learning project that predicts house prices using real estate features.

## 📌 Objective

To build a machine learning model that predicts house prices based on:

- Area
- Bedrooms
- Bathrooms
- Location
- House age
- Parking spaces

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## 🤖 Machine Learning Model

- Random Forest Regressor

## 📂 Project Structure

```text
real-estate-house-price-prediction-ml/
│
├── data/
│   └── house_data.csv
│
├── models/
│   └── house_price_model.pkl
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict_price.py
│   └── evaluate_model.py
│
├── outputs/
│   └── prediction_results.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
````

## 🚀 How to Run

### 1. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train model

```bash
python src/train_model.py
```

### 4. Evaluate model

```bash
python src/evaluate_model.py
```

### 5. Run web app

```bash
streamlit run app.py
```

## 📊 Sample Prediction

Input:

```text
Area: 2200 sqft
Bedrooms: 3
Bathrooms: 2
Location: Colombo
House Age: 5
Parking: 1
```

Output:

```text
Predicted House Price: 520000
```

## 📈 Future Improvements

* Add larger real-world dataset
* Add data visualization
* Add model comparison
* Deploy Streamlit app online

````

---

## Run order

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
python src/evaluate_model.py
streamlit run app.py
````


