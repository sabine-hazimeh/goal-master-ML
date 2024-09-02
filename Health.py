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