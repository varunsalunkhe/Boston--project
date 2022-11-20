<<<<<<< HEAD
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, app,  render_template
from flask import request

app =Flask(__name__)

model = joblib.load("model.pkl")

scale =joblib.load("scale.pkl")

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/boston-predict', methods=['GET','POST'])

def predict_api():
    if model:
        try:
            data = request.json
                        
            querry=pd.DataFrame(data)
            output=model.predict(scale.transform(querry))
            print(output)
            return jsonify({"prize" : str(output)})
        
        except:
            return jsonify({"trace ": traceback.format_exc()})

    else:
        print("first train the model")
        return ("no model is here to use")
		
           
@app.route("/predict", methods =["POST", "GET"])
def predict():
    data= [float(x) for x in request.form.values()]
    final_input= scale.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    return render_template('home.html', Prize ="The predicted prize is {}".format(round(output,4)))


if __name__ == "__main__":
=======
import joblib
import pandas as pd
import numpy as np
from flask import Flask, jsonify, app,  render_template
from flask import request

app =Flask(__name__)

model = joblib.load("model.pkl")

scale =joblib.load("scale.pkl")

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/boston-predict', methods=['GET','POST'])

def predict():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scale.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__ == "__main__":
>>>>>>> 5bae35f920fdee6593aedc2c61031c54d3e15a90
	app.run(debug= True)