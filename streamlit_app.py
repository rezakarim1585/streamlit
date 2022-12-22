
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import math
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
            'pf':pf,
            'Ef (GPA)':Ef
           }

    data_log = {'bw': np.log(bw),
            'd': np.log(d),
            #"fc": np.log(fc),
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



import pickle
import math


rf_model1 = pickle.load(open('rf_model.pkl', 'rb'))

xb_model1 = pickle.load(open('xb_model.pkl','rb'))


rf_model = RandomForestRegressor()
rf_model.fit(X1, Y)
# Apply Model to Make Prediction
xb_model = XGBRegressor()
xb_model.fit(X1, Y)

prediction = rf_model.predict(df1)

prediction2 = xb_model.predict(df1)

prediction3 = rf_model1.predict(df1)


prediction4 = xb_model1.predict(df1)


st.header('Prediction of Vexp')
st.write(prediction,prediction2)
st.write(prediction3,prediction4)
st.write('---')




