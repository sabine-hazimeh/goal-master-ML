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
    df['sentiment_score'] = df['emotion'].map(emotion_mapping)
    df['created_at'] = pd.to_datetime(df['created_at'])

    sentiment_over_time = df.groupby(df['created_at'].dt.date)['sentiment_score'].mean().reset_index()
    sentiment_over_time.columns = ['date', 'average_sentiment']

    return jsonify(sentiment_over_time.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(port=5002)

