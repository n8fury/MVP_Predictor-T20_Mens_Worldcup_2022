from flask import Flask, request, jsonify, render_template, url_for
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
with open('scaler.pkl', 'rb') as f:
    scalerO = pickle.load(f)
