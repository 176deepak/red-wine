import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        quality = None
        fa = float(request.form['fixed-acidity'])
        va = float(request.form['volatile-acidity'])
        rs = float(request.form['residual-sugar'])
        cl = float(request.form['chlorides'])
        f_so2 = float(request.form['free-sulfur-dioxide'])
        t_so2 = float(request.form['total-sulfur-dioxide'])
        density = float(request.form['density'])
        pH = float(request.form['pH'])
        sulphates = float(request.form['sulphates'])
        alcohol = float(request.form['alcohol'])
        
        data = {
            'fixed acidity': [fa],
            'volatile acidity': [va],
            'residual sugar': [rs],
            'chlorides': [cl],
            'free sulfur dioxide': [f_so2],
            'total sulfur dioxide': [t_so2],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol]
        }

        cols = list(data.keys())

        data = pd.DataFrame(data=data)
        preprocessor = pickle.load(open('Models\Saved_Models\preprocessor.pkl', 'rb'))
        data_scaled=preprocessor.transform(data)
        data = pd.DataFrame(data_scaled, columns=cols)

        predictor = pickle.load(open('Models\Saved_Models\model.pkl', 'rb'))
        pred = predictor.predict(data)
        print(pred)

        return render_template('index.html', quality = quality)

if __name__ == '__main__':
    app.run(debug=True)