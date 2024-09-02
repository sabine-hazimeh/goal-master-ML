from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from flask_cors import CORS

warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app) 

df = pd.read_csv('C:/Users/Admin/Downloads/final_dataset.csv')

df = df.dropna()

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

columns_to_check = ['Age', 'Height', 'Weight', 'BMI']
for column in columns_to_check:
    df = remove_outliers(df, column)

df['Exercise Plan Description'] = df['Exercise Recommendation Plan'].map({
    1: "Light Exercise",
    2: "Moderate Exercise",
    3: "Intense Exercise",
    4: "Cardio Focused",
    5: "Strength Training",
    6: "Flexibility Training",
    7: "Comprehensive Plan"
})

df['Height_to_Weight_Ratio'] = df['Height'] / df['Weight']
df['BMI_Age_Interaction'] = df['BMI'] * df['Age']

df_encoded = pd.get_dummies(df, columns=['Gender', 'BMIcase'])

X = df_encoded.drop(['Exercise Recommendation Plan', 'Exercise Plan Description'], axis=1)
y = df_encoded[['Exercise Recommendation Plan']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor()
model.fit(X_train, y_train)
joblib.dump(model, 'model.pkl')

model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    age = data['age']
    gender = data['gender']
    height = data['height']
    weight = data['weight']

    input_df = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Height': [height],
        'Weight': [weight],
        'BMI': [weight / (height/100)**2],
        'BMIcase': ['Normal' if weight / (height/100)**2 < 25 else 'Overweight'],
        'Height_to_Weight_Ratio': [height / weight],
        'BMI_Age_Interaction': [(weight / (height/100)**2) * age]
    })

    input_encoded = pd.get_dummies(input_df, columns=['Gender', 'BMIcase'])
    input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

    prediction = model.predict(input_encoded)[0]
    predicted_plan_code = round(prediction)

    exercise_details = {
        1: {
            "name": "Light Exercise",
            "explanation": "Light exercise includes activities like walking or light stretching.",
            "ideal_times": "Ideal to perform in the morning before breakfast or in the evening after work."
        },
        2: {
            "name": "Moderate Exercise",
            "explanation": "Moderate exercise includes activities like brisk walking or light jogging.",
            "ideal_times": "Best performed in the morning or early evening."
        },
        3: {
            "name": "Intense Exercise",
            "explanation": "Intense exercise includes activities like running, HIIT, or heavy weight lifting.",
            "ideal_times": "Preferably done in the morning when your energy levels are high."
        },
        4: {
            "name": "Cardio Focused",
            "explanation": "Cardio exercises like running, cycling, or swimming that increase your heart rate.",
            "ideal_times": "Ideal to perform early in the morning on an empty stomach."
        },
        5: {
            "name": "Strength Training",
            "explanation": "Strength training includes exercises like weight lifting or bodyweight exercises.",
            "ideal_times": "Best performed in the afternoon or evening after meals."
        },
        6: {
            "name": "Flexibility Training",
            "explanation": "Flexibility training includes yoga, pilates, or stretching exercises.",
            "ideal_times": "Ideal for morning or before bed to relax your body."
        },
        7: {
            "name": "Comprehensive Plan",
            "explanation": "A balanced plan including cardio, strength, and flexibility exercises.",
            "ideal_times": "Can be spread out throughout the day, with different types of exercises at different times."
        }
    }

    predicted_plan = exercise_details.get(predicted_plan_code, {
        "name": "Unknown Plan",
        "explanation": "No information available for this plan.",
        "ideal_times": "No suggested times available."
    })

    return jsonify({
        'predicted_plan': predicted_plan['name'],
        'explanation': predicted_plan['explanation'],
        'ideal_times': predicted_plan['ideal_times']
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)