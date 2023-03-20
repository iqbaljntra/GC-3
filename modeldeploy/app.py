import streamlit as st
import pandas as pd
import numpy as np
import joblib

with open('model_rf_model.pkl', 'rb') as file_1:
  model_rf_model= joblib.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler=joblib.load(file_2)

with open('list_num_cols.txt', 'rb') as file_4:
  num_cols= joblib.load(file_4)

with open('list_cat_cols.txt', 'rb') as file_5:
  cat_cols= joblib.load(file_5)


age = st.slider('Masukan Umur:',0, 42, step=1)
creatinine_phosphokinase = st.number_input('Masukan total creatinine phosphokinase :')
anaemia = st.radio('Apakah anda mengidap Anaemia? 1=ya, 0=tidak',(0, 1))
serum_creatinine = st.slider('Masukan Serum Creatinine:',0.0, 10.0)
high_blood_pressure = st.radio('Apakah anda mengidap Hipertensi? 1=ya, 0=tidak',(0, 1))
smoking = st.radio('Apakah anda merokok? 1=ya, 0=tidak',(0, 1))


if st.button('Predict'):

    data_inf = pd.DataFrame({
    'age': age,
    'creatinine_phosphokinase': creatinine_phosphokinase,
    'anaemia': anaemia,
    'serum_creatinine':serum_creatinine,
    'high_blood_pressure': high_blood_pressure,
    'smoking':smoking },index=[0])
    
    data_inf_scaled = model_scaler.transform(data_inf[num_cols])
    data_inf_fix = np.concatenate([data_inf_scaled, data_inf[cat_cols]], axis=1)
    hasil = model_rf_model.predict(data_inf_fix)
    pred =''
    if hasil == 0 :
        pred = 'Anda Sehat'
    else:
        pred = 'Anda Tidak Lama Lagi Terkena Serangan Jantung'
    st.header(f'Death Event= {pred}')



