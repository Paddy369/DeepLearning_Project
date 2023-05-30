from flask import Flask, jsonify, request
import os
import re
from tensorflow.keras import models
from predict import classify

app = Flask(__name__)

@app.route('/api/milestones', methods=['GET','POST','OPTIONS'])
def fetch_milestones():
    pattern = re.compile("Meilenstein[a-zA-Z0-9]*")
    milestones = [name for name in os.listdir("../..") if pattern.match(name)]

    response = jsonify(milestones)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/models', methods=['GET','OPTIONS'])
def fetch_models():
    milestone = request.args.get('milestone')
    models = os.listdir("../../" + milestone + "/saved_models")

    response = jsonify(models)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/classify', methods=['GET','OPTIONS'])
def fetch_results():
    image = request.args.get('image')
    milestone = request.args.get('milestone')
    modelName = request.args.get('model')

    results = classify(image, milestone, modelName)

    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response