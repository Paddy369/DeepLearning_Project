from flask import Flask, jsonify, request
import os
import re

app = Flask(__name__)

@app.route('/api/milestones', methods=['GET','POST','OPTIONS'])
def fetch_milestones():
    pattern = re.compile("Meilenstein[a-zA-Z0-9]*")
    milestones = [name for name in os.listdir("../..") if pattern.match(name)]

    response = jsonify(milestones)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/api/models', methods=['GET','POST','OPTIONS'])
def fetch_models():
    milestone = request.args.get('milestone')
    models = os.listdir("../../" + milestone + "/saved_models")

    response = jsonify(models)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response