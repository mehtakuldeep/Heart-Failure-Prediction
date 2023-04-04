import numpy as np
from flask import Flask, render_template,request,redirect
import pickle
import os
#Initialize the flask App

app = Flask(__name__, template_folder='templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET','POST'])
def home():
    
   return render_template("home.html")

@app.route('/index.html', methods = ['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/index.html/predict',methods=['POST'])
def predict():
    age = request.form['age']
    sex = request.form.get('sex')
    cp = request.form.get('cp')
    trtbps = request.form['trtbps']
    chol = request.form['chol']
    fbs = request.form.get('fbs')
    restecg = request.form.get('restecg')
    thalachh = request.form['thalachh']
    exng = request.form.get('exng')
    oldpeak = request.form['oldpeak']
    slp = request.form['slp']
    caa = request.form['caa']
    thall = request.form['thall']

    age = int(age)
    sex = int(sex)
    trtbps = int(trtbps)
    chol = int(chol)
    fbs = int(fbs)
    restecg = int(restecg)
    thalachh = int(thalachh)
    exng = int(exng)
    oldpeak = float(oldpeak)
    slp = int(slp)
    caa = int(caa)
    thall = int(thall)
    
    final_features = np.array([(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)])

    prediction = model.predict_proba(final_features)
    
    return render_template("index.html", prediction_text='Percentage of getting a heart attack: {}'.format(prediction[0][1] * 100))

if __name__ == "__main__":
    app.run(debug=True)