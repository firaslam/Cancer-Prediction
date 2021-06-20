from flask import Flask,request,render_template
import pandas as pd #python library used to analyse data
import numpy as np # python library used for working  with arrays
import pickle
from utility import PandasSelector


app = Flask(__name__)

# saved the pre-trained model in pickle format(pickle convert python object to byte stream)
pipe = pickle.load(open('pipe.pkl','rb'))  #pickle.load for saving , open () the  model wich got picked , rb to open it with reading bytes

feature_cols = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean']

# create "list" :
#radius (mean of distances from center to points on the perimeter)
#texture (standard deviation of gray-scale values)
#smoothness (local variation in radius lengths)
# compactness (perimeter^2 / area - 1.0)

@app.route("/")
def getModel():
    return render_template("form.html") # the form.html file should be in the `templates` subfolder


@app.route('/predict', methods=["POST"])
def predict():

    input_data = []
    for col in feature_cols:
        input_data.append(float(request.form[col]))

    input_df = pd.DataFrame(np.array(input_data).reshape(1,-1), columns=feature_cols)

    loaded_model = pipe.steps[1][1]
    processing = pipe.steps[0][1]
    X = processing.transform(input_df)
    pred = loaded_model.predict(X)

    return str(pred[0])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True) 
