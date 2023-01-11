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
    rbwi = round(bowling_impact, 2)
    total_impact = rbi + rbwi
    if keyplayer == 'Yes':
        kp = 1
    else:
        kp = 0

    features = [totalmatches, totalrun, battingavg, strikerate, wickets, economy,
                rbi, rwpm, rbwi, total_impact, sm, kp]
    input_data = np.array(features)
    input_data_reshaped = input_data.reshape(1, -1)
    final_features = scalerO.transform(input_data_reshaped)
    prediction = model.predict(final_features)
    print("features", features)
    print("final_features", final_features)
    print("prediction", prediction)
    output = round(prediction[0], 2)
    print(output)
    if output == 0:
        return render_template('index.html', prediction_text='Player can not be MVP')
    else:
        return render_template('index.html', prediction_text='Player can be  MVP')
    


@app.route('/predict_api', methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=False)
