
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
#from sklearn.neighbors import KNeighborsRegressor


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
    bw = st.sidebar.slider('Width of the beam, **bw(mm)**', float(X.bw.min()), float(X.bw.max()), float(X.bw.mean()))
    d = st.sidebar.slider('Effective depth of the beam, **d (mm**)', float(X.d.min()), float(X.d.max()), float(X.d.mean()))
    fc = st.sidebar.slider("Compressive strength of concrete, **fc'(MPa)**", float(X.fc.min()), float(X.fc.max()), float(X.fc.mean()))
    pf = st.sidebar.slider('Reinforcement ratio, **ρ**', float(X.pf.min()), float(X.pf.max()), float(X.pf.mean()))
    Ef= st.sidebar.slider('Modulas of Elasticity of Rebar, **Ef (GPA)**', float(X.Ef.min()), float(X.Ef.max()), float(X.Ef.mean()))
    ad= st.sidebar.slider('Shear span-to-depth ratio, **a/d**', float(X.a_d.min()), float(X.a_d.max()), float(X.a_d.mean()))                   
    data = {'bw(mm)': bw,
            'd (mm)': d,
            "fc' (MPa)": fc,
            'a/d': ad,
            'ρf':pf,
            'Ef (GPA)':Ef
           }
    df2 = pd.DataFrame(data, index=[0])
    st.subheader('User Input parameters')
    st.table(df2)
    
    data_log = {'bw': np.log(bw),
            'd': np.log(d),
            "fc": np.log(fc),
            #"fc": np.log(np.sqrt(fc)),
            'a_d': np.log(ad),
            'pf':pf,
            'Ef':np.log(Ef)
           }                    

    features = pd.DataFrame(data_log, index=[0])
    return features

col1,col2=st.columns([1,1])
with col1:
   df1 = user_input_features()
with col2: 
 image = Image.open('pic1.png')
 st.image(image, use_column_width=True)

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

import pickle
import math

lr_model = pickle.load(open('lr_model.pkl', 'rb'))
rr_model = pickle.load(open('rr_model.pkl','rb'))
lasr_model = pickle.load(open('lasr_model.pkl', 'rb'))
dt_model = pickle.load(open('dt_model.pkl','rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
svr_model = pickle.load(open('svr_model.pkl','rb'))
ab_model = pickle.load(open('ab_model.pkl', 'rb'))
xb_model = pickle.load(open('xb_model.pkl','rb'))
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
ann_model = pickle.load(open('ann_model.pkl','rb'))



p_lr = float(lr_model.predict(df1))
p_rr = float(rr_model.predict(df1))

p_lasr = float(lasr_model.predict(df1))
p_svr = float(svr_model.predict(df1))

p_dt = float(dt_model.predict(df1))
p_rf = float(rf_model.predict(df1))

p_xb = float(xb_model.predict(df1))
p_ab= float(ab_model.predict(df1))

p_knn = float(knn_model.predict(df1))
p_ann = float(ann_model.predict(df1))


st.header('Prediction of Vexp')



pred = { 'ML Algorithm': ['Multilinear Regression (LR)','Ridge Regression (RR)','Lasso Regression (LaR)','Support Vector Regression (SVR)','Decision Tree (DT)','Random Forest (RF)','K-Nearest Neighbour (KNN)','Arti-Neural Network (ANN)','Adaboost (AB)','Extreme Grad- Boost (XB)'],
         'Predicted Shear Strength (KN)': [p_lr, p_rr, p_lasr, p_svr,p_dt,p_rf, p_knn,p_ann, p_ab,p_xb]
       }
 
df_pred = pd.DataFrame(pred)
 
st.write(df_pred)

st.line_chart(data=df_pred, x='ML Algorithm', y='Predicted Shear Strength (KN)', width=0, height=0, use_container_width=True)
#st.write(p_lr,p_rr,p_lasr,p_svr,p_dt,p_rf,p_xb,p_ab,p_knn,p_ann)
st.write('---')













