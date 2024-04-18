import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from flask import Flask,request,jsonify,render_template

application=Flask(__name__)
app=application

# import ridge regressor model and standard scaler pickle
best_model=pickle.load(open('best_model.pkl', 'rb'))
standard_scaler=pickle.load(open('scaler.pkl', 'rb'))

# route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get', 'Post'])
def predict_datapoint():
    if request.method=='POST':
        PV=float(request.form.get('payment_value'))
        TTD=float(request.form.get('time_to_delivery'))
        EAS=float(request.form.get('estimated_vs_actual_shipping'))
        LD=float(request.form.get('late_delivery'))

        # new_data_sc=standard_scaler.transform([[PV,TTD,EAS,LD]])
        new_data_sc=[PV,TTD,EAS,LD]
        result= best_model.predict(new_data_sc)

        return render_template('index.html', result=result[0])
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)