from flask import Flask, jsonify
import os
import re

app = Flask(__name__)

@app.route('/api/milestones', methods=['GET','POST','OPTIONS'])
def hello_world():
    pattern = re.compile("Meilenstein[a-zA-Z0-9]*")
    milestones = [name for name in os.listdir("../..") if pattern.match(name)]

    response = jsonify(milestones)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

