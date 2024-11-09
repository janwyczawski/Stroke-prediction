import pandas as pd
from flask import Flask, request, jsonify
import joblib

model = joblib.load("logistic_regression_tuned.pkl")

app = Flask(__name__)
column_names = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data["features"], columns=column_names)
    predictions = model.predict(df)

    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)