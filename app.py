import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
# initialize our Flask application 
app = Flask(__name__,static_url_path='/static')
# loading model
model = pickle.load(open('model.pkl', 'rb'))
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    # final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    y_probabilities_test = model.predict_proba(final_features)
    y_prob_success = y_probabilities_test[:, 1]
    print("final features",final_features)
    print("prediction:",prediction)
    output = round(prediction[0], 2)
    y_prob=round(y_prob_success[0], 3)
    print(output)
# ouputs the probability of success
    if output == 0:
        return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE{}'.format(y_prob))
    else:
         return render_template('index.html', prediction_text='THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER {}'.format(y_prob))
# create a route for the default page for json
@app.route('/predict_api',methods=['POST'])
def predict_api():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
