import joblib
import pandas as pd
import numpy as np
import flask as Flask, app, request, jsonify, url_for, render_template

app =Flask(__name__)

model = joblib.load("model.pkl")

scale =joblib.load("scale.pkl")

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/boston-predict', methods=['GET','POST'])

def predict:
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scale.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
	app.run(debug= True)