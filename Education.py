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