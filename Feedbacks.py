from flask import Flask, jsonify
import pandas as pd
import mysql.connector
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/sentiment', methods=['GET'])
def get_sentiment_data():
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",
        database="goal_master_db"
    )
    query = "SELECT emotion, created_at FROM emotions WHERE type = 'detected'"
    df = pd.read_sql(query, conn)
    conn.close()
    emotion_mapping = {
        'happy': 2,
        'neutral': 0,
        'sad': -1,
        'angry': -2,
        'surprised': 1
    }

