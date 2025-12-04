from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("python-ml/model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    result = {
        "fraud_probability": float(probability),
        "decision": "BLOCK" if prediction == 1 else "ALLOW"
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
