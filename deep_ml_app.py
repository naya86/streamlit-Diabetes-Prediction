import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from sklearn.ensemble import RandomForestClassifier
import joblib



def deep_run_ml_app():
    st.subheader('예측하기')
    st.write('당뇨병을 예측하기 위해선 데이터 입력이 필요합니다.')
    
    pregnancies = st.number_input('Pregnancies 입력', min_value=1)
    glucose = st.number_input('Glucose 입력', min_value=1)
    bloodPressure = st.number_input('BloodPressure 입력', min_value=1)
    skinThickness = st.number_input('SkinThickness 입력', min_value=1)
    insulin = st.number_input('Insulin 입력', min_value=1)    
    bmi = st.number_input('BMI 입력', min_value=1)
    diabetesPedigreeFunction= st.number_input('DiabetesPedigreeFunction 입력', min_value=1)
    age = st.number_input('Age 입력', min_value=1)

    new_model = tensorflow.keras.models.load_model('data/deep check(new).h5')

    new_data = np.array( [ [ pregnancies, glucose, bloodPressure, skinThickness, insulin,bmi, diabetesPedigreeFunction, age  ] ] ) ## 2차원 설정.

    sc_X = joblib.load('data/sc_X.pkl')
    new_data = sc_X.transform(new_data)
    
    
    

    # 예측.

    y_pred = new_model.predict(new_data)
    
    if y_pred==0 :
        y_pred = '당뇨병이 아닙니다.'
        
    elif y_pred == 1:
        y_pred = '당뇨병입니다.'
           
    
    
    if st.button('결과 확인') :

        st.write('예측결과는 {}'.format(y_pred))





    








    
