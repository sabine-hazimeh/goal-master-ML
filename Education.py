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
    knn = NearestNeighbors(n_neighbors=20, algorithm='auto')
    knn.fit(features_scaled)
    user_level_encoded = label_encoder.transform([current_level])[0]
    user_input_df = pd.DataFrame(
        [[total_hours, user_level_encoded]],
        columns=['Duration to complete (Approx.)', 'Level_encoded']
    )
    user_input_scaled = scaler.transform(user_input_df)
    distances, indices = knn.kneighbors(user_input_scaled)
    recommended_courses = df_filtered.iloc[indices[0]]

    if not recommended_courses.empty:
        result = recommended_courses[['Course Title', 'Duration to complete (Approx.)', 'Level', 'Course Url']].to_dict(orient='records')
        return jsonify({"recommended_courses": result})

    else:
        return jsonify({"message": "No courses fit within the available hours. Consider adjusting your plan."}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
