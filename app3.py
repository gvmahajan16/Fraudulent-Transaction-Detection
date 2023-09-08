import xgboost
from flask import Flask,render_template
import requests
import pickle
import pandas as pd
import numpy as np
import sklearn
import lightgbm
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import  MinMaxScaler,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder, PowerTransformer


app = Flask(__name__)



with open("model2.pkl", 'rb') as file:  XGBM = pickle.load(file)
with open("pipe1.pkl", 'rb') as file1:  preprocessor = pickle.load(file1)
print(preprocessor)


     

@app.route("/")
def Index():
     return render_template('Index3.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    
    
    file = request.files['file']
    fileName = file.filename
    file.save(secure_filename(file.filename))
    df=pd.read_csv(fileName)
    x_test=df
    print(df.shape)
    x_n = preprocessor.transform(x_test)
    print(x_n.shape)
    y_test_pred_lgbm = XGBM.predict_proba(x_n)[:,1]
    
    #make prediction
    # prediction=""
    # for i in y_test_pred_lgbm:
    #   if i > 0.5:
    #      prediction="Transaction is fraud"
    #   else:
    #      prediction="Transaction is non-fraud"
    #
    for i in y_test_pred_lgbm:
        if i > 0.5:
            status_text = "This Transaction is Fraud"
        else:
            status_text = "This Transaction is NOT Fraud"
    print(status_text)
    print(df.shape)
    # return render_template('Index.html', prediction_text='This {} transaction.'.format(prediction))
    return render_template('Index3.html', prediction_result=status_text)


if __name__=='__main__':
    app.run(debug=True)
