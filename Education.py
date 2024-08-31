from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)  

df = pd.read_csv(r'C:/Users/Admin/Desktop/Education_Cleaned.csv')

label_encoder = LabelEncoder()
df['Level_encoded'] = label_encoder.fit_transform(df['Level'])

def calculate_available_hours(start_date, end_date, daily_hours, weekly_days):
    num_days = (end_date - start_date).days
    if num_days < 0:
        raise ValueError("Target date must be after the current date")
    
    full_weeks = num_days // 7
    remaining_days = num_days % 7
    
    total_hours = (full_weeks * weekly_days + min(remaining_days, weekly_days)) * daily_hours
    return total_hours
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    desired_course = data.get('desired_course')
    available_hours_per_day = data.get('available_hours_per_day')
    available_days_per_week = data.get('available_days_per_week')
    target_date_str = data.get('target_date')
    current_level = data.get('current_level')
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    current_date = datetime.now()
    try:
        total_hours = calculate_available_hours(current_date, target_date, available_hours_per_day, available_days_per_week)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    df_filtered = df[df['Course Title'].str.contains(desired_course, case=False, na=False)]
    features = df_filtered[['Duration to complete (Approx.)', 'Level_encoded']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)