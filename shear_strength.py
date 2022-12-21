#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
#from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(layout="wide", page_title=None, page_icon=":house:")

st.write("""
 ## Shear Strength Prediction of Slender Concrete Beams Reinforced with FRP Rebar using Data-driven Machine Learning Algorithms
**Mohammad Rezaul Karim, Kamrul Islam, AHM Muntasir Billah, M. Shahria Alam** 
""")
#st.latex(r''' e^{i\pi} + 1 = 0 ''')

df = pd.read_csv('ssdata.csv')


df= df[['bw']+['d']+['fc']+['a_d']+['pf']+['Ef']+['Vexp']]

df=pd.DataFrame(df)
#st.write(df)
X = df.loc[:, df.columns != 'Vexp']
st.sidebar.header('User Input Parameters')

def user_input_features():
    bw = st.sidebar.slider('**bw(mm)**', float(X.bw.min()), float(X.bw.max()), float(X.bw.mean()))
    d = st.sidebar.slider('**d (mm**)', float(X.d.min()), float(X.d.max()), float(X.d.mean()))
    fc = st.sidebar.slider("**fc'(MPa)**", float(X.fc.min()), float(X.fc.max()), float(X.fc.mean()))
    pf = st.sidebar.slider('**pf**', float(X.pf.min()), float(X.pf.max()), float(X.pf.mean()))
    Ef= st.sidebar.slider('**Ef (GPA)**', float(X.Ef.min()), float(X.Ef.max()), float(X.Ef.mean()))
    ad= st.sidebar.slider('**a/d**', float(X.a_d.min()), float(X.a_d.max()), float(X.a_d.mean()))                   
    data = {'bw(mm)': bw,
            'd (mm)': d,
            "fc' (MPa)": fc,
            'a/d': ad,
            'pf':pf,
            'Ef (GPA)':Ef
           }
    #data1=pd.DataFrame(data, index=[0])
    #st.write(data1)
    data_log = {'bw': np.log(bw),
            'd': np.log(d),
            "fc": np.log(np.sqrt(fc)),
            'a_d': np.log(ad),
            'pf':pf,
            'Ef':np.log(Ef)
           }                    

    features = pd.DataFrame(data_log, index=[0])
    return features

df1 = user_input_features()
st.subheader('User Input parameters')
st.write(df1)



df['bw'] = np.log(df['bw'])
df['d'] = np.log(df['d'])
df['fc'] = np.log(df['fc'])
df['a_d'] = np.log(df['a_d'])
#df['pf'] = np.log(df['pf'])
df['Ef'] = np.log(df['Ef'])
#df['Vexp'] = np.log(df['Vexp'])

df=pd.DataFrame(df)
#st.write(df)

X1 = df.loc[:, df.columns != 'Vexp']
Y=df['Vexp']


#from flask import Flask, request, jsonify, render_template
import pickle
from numpy import inf
import math

#app = Flask(__name__,static_url_path = "/tmp", static_folder = "tmp")
#rf_model = pickle.load(open('fprc_r_rf.pkl', 'rb'))
ab_model = pickle.load(open('fprc_r_ab.pkl', 'rb'))
ann_model = pickle.load(open('fprc_r_ann.pkl','rb'))
#cb_model = pickle.load(open('fprc_r_cb.pkl','rb'))
dt_model = pickle.load(open('fprc_r_dt.pkl','rb'))
#knn_model = pickle.load(open('knn_model.pkl','rb'))
lasso_model = pickle.load(open('fprc_r_lasso.pkl','rb'))
lr_model = pickle.load(open('fprc_r_lr.pkl','rb'))
ridge_model = pickle.load(open('fprc_r_rr.pkl','rb'))
svr_model = pickle.load(open('svr_model.pkl','rb'))
#xg_model = pickle.load(open('fprc_r_xb.pkl','rb'))


#scaler = pickle.load(open('scaler.pkl', 'rb'))
#scaler.clip = False
#df1=scaler.transform(df1)
#df1 = [np.array(df1)]


# Build Regression Model

#rf_prediction = rf_model.predict(final_features)

rf_model = RandomForestRegressor()
rf_model.fit(X1, Y)
# Apply Model to Make Prediction
xb_model = XGBRegressor()
xb_model.fit(X1, Y)

prediction = rf_model.predict(df1)
#prediction=math.exp(prediction[0])

prediction2 = xb_model.predict(df1)
#prediction2=math.exp(prediction2[0])

prediction3 = svr_model.predict(df1)
#prediction3=math.exp(prediction3[0])

prediction4 = ab_model.predict(df1)
#prediction4=math.exp(prediction4[0])

prediction5 = ridge_model.predict(df1)
#prediction5=math.exp(prediction5[0])

prediction6 = dt_model.predict(df1)
#prediction6=math.exp(prediction6[0])

st.header('Prediction of Vexp')
st.write(prediction,prediction2, prediction3,prediction4, prediction5,prediction6)

st.write('---')




