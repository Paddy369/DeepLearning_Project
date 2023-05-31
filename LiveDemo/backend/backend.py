from flask import Flask, jsonify, request
import os
import re
from predict import classify

app = Flask(__name__)

# endpoint to return all available milestones
@app.route('/api/milestones', methods=['GET','POST','OPTIONS'])
def fetch_milestones():
    # pattern to match the milestone names
    pattern = re.compile("Meilenstein[a-zA-Z0-9]*")
    # get all directories with a matching name
    milestones = [name for name in os.listdir("../..") if pattern.match(name)]

    # convert to json, allow cross origin requests and return
    response = jsonify(milestones)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# endpoint to return all available models for a given milestone
@app.route('/api/models', methods=['GET','OPTIONS'])
def fetch_models():
    milestone = request.args.get('milestone')
    # load all models from the saved_models directory for the given milestone
    models = os.listdir("../../" + milestone + "/saved_models")

    # convert to json, allow cross origin requests and return
    response = jsonify(models)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# endpoint to return the results of a classification for a given image, milestone and model
@app.route('/api/classify', methods=['GET','OPTIONS'])
def fetch_results():
    image = request.args.get('image')
    milestone = request.args.get('milestone')
    modelName = request.args.get('model')

    # create a demo directory if it does not exist
    os.makedirs("./demo", exist_ok=True)
    # classify the image 
    results = classify(image, milestone, modelName)

    # convert to json, allow cross origin requests and return
    response = jsonify(results)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response