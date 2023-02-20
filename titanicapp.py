import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()

st.title("Titanic Survival Prediction using Random Classifier")
Pclass = st.selectbox("Please select a class",[1,2,3])
Sex1 = st.selectbox("Please select a gender",["Male","Female"])
Sex = 1 if Sex1 == 'Male' else 0 
SibSp = st.slider("Number of siblings / spouses aboard the Titanic",0,10)
Parch = st.slider("Number of parents / children aboard the Titanic",0,2)

df_pred = pd.DataFrame([[Pclass,Sex,SibSp,Parch]])
df_pred

model = joblib.load('titanic_rf_model.pkl')
prediction = model.predict(df_pred)

def predict():
    if prediction[0] == 1: 
        st.success('Passenger survived :thumbsup:')
    else: 
        st.error('Passenger did not survive :thumbsdown:') 

trigger = st.button('Predict', on_click=predict)