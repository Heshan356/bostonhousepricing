from distutils.log import debug
from json.tool import main
import pickle
from urllib import request
from django.urls import path
from django.http import HttpResponse
from django.template import loader
import numpy as np
import json
from django import apps
from flask import *

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler = pickle.load(open('scalling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data) 
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    out_put = regmodel.predict(new_data)
    print(out_put[0])
    return jsonify(out_put[0])

if __name__=="__main__":
    app.run(debug=True)