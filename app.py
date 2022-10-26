from distutils.log import debug
from json.tool import main
import pickle
from urllib import request
import numpy as np
import json
from django import apps
from flask import *

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
Scaler = pickle.load(open('scalling.pkl','rb'))
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

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=Scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)