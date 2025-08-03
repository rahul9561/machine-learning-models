from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  # <-- Import CORS
from utils.preprocess import preprocess_input

app = Flask(__name__)
CORS(app)  # <-- Enable CORS for all routes

# Load Model and Scaler
model = joblib.load('model/loan_model.joblib')
scaler = joblib.load('model/scaler.joblib')


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'UP'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        processed = preprocess_input(input_data)
        scaled = scaler.transform(processed)
        prediction = model.predict(scaled)[0]
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
