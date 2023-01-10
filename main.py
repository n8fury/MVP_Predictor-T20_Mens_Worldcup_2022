from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
with open('scaler.pkl', 'rb') as f:
    scalerO = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    totalmatches = float(request.form['totalmatches'])
    totalrun = float(request.form['totalrun'])
    battingavg = float(request.form['battingavg'])
    strikerate = float(request.form['strikerate'])
    wickets = float(request.form['wickets'])
    economy = float(request.form['economy'])
    semiprobability = float(request.form['semiprobability'])




@app.route('/predict_api', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=False)
