import sys
import os
import inspect
import requests

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from waitress import serve
from flask import Flask, jsonify, make_response, request
from utils import load_models, predict

app = Flask(__name__)

models_path = parentdir + "/trained_models"
model_labels = []
models = []


@app.route("/health")
def health_check():
	return_str = "ROLF is running"
	returnJSON = jsonify(Message = return_str)
	return returnJSON

@app.route('/predict_text', methods=['POST'])
def predict_text():
    text = request.form.get('text')

    if text is None:
        return jsonify({'error': 'No text provided'}), 400

    load_models(models_path, model_labels, models)
    probs, labels = predict(model_labels, models, text)
    json_obj =  jsonify({'probs': probs, 'labels': labels})
    return json_obj


@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.form.get('url')

    if url is None:
        return jsonify({'error': 'No url provided'}), 400

    response = requests.get(url)
    text = response.text
    load_models(models_path, model_labels, models)
    probs, labels = predict(model_labels, models, text)
    json_obj =  jsonify({'probs': probs, 'labels': labels})
    return json_obj


if __name__ == "__main__":
	print("Starting ROLF on port 8080")
	serve(app, host="0.0.0.0", port=8080)
