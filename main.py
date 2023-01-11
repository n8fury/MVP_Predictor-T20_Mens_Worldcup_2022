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
    sm = semiprobability / 100.00
    keyplayer = request.form['keyplayer']
    batting_impact = ((battingavg*strikerate)/160.00)/100.00
    rbi = round(batting_impact, 2)
    wicketpermatch = wickets/totalmatches
    rwpm = round(wicketpermatch, 2)
    ew = economy*wicketpermatch
    if ew == 0:
        bowling_impact = 0.00
    else:
        bowling_impact = (96.00/(ew))/100.00



@app.route('/predict_api', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=False)
