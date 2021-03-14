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
from deep_ml_app import deep_run_ml_app



def eda_run_app():
    st.write('데이터 정보')
    st.write(''' Pregnancies : Number of times pregnant 임신횟수

    Glucose : 공복혈당 Plasma glucose concentration a 2 hours in an oral glucose tolerance test

    BloodPressure : Diastolic blood pressure (mm Hg)

    SkinThickness : Triceps skin fold thickness (mm)

    Insulin : 2-Hour serum insulin (mu U/ml)

    BMI : Body mass index (weight in kg/(height in m)^2)

    Diabetes pedigree function

    Age (years)

    COutcome : class variable (0 or 1) 268 of 768 are 1, the others are 0''')
    
    df = pd.read_csv('data/diabetes.csv')
    selected_radio = st.radio('원하는 정보 선택', ['DataFrame','통계치', '상관관계분석', '최대값,최소값'])
    
    
    if selected_radio == 'DataFrame' :
        st.dataframe(df)
        st.write('Outcome 1 = 당뇨병')
        columns = df.columns
        columns = list(columns)
        selected_columns = st.multiselect('원하는 컬럼을 선택하세요.', columns)

        if len(selected_columns) > 0 :
            st.dataframe( df[ selected_columns ])

    elif selected_radio == '통계치' :
        st.dataframe(df.describe())   

    elif selected_radio == '상관관계분석' :
        st.subheader('상관관계분석은 -1부터 1까지의 숫자표기로, 1로 가까워질수록 비례하는 값을 갖는다.(관여율이 높다)')
        st.dataframe(df.corr())

        corr_columns = df.columns [df.dtypes != object ]
        selected_corr = st.multiselect('상관계수 컬럼선택', corr_columns)         

        if len(selected_corr) > 0 :
            st.dataframe( df[ selected_corr ].corr() )         ## 위에서 선택한 컬럼들을 이용해서, pairplot 차트 그리기.
   
            if st.button('Chart'):
                fig = sns.pairplot(data = df[selected_corr].corr())     
                st.pyplot(fig) 

    elif selected_radio == '최대값,최소값' :
        number_columns = df.columns [df.dtypes != object ]
        min_max = st.selectbox('컬럼을 선택하세요.', number_columns)
    
        if len(min_max) != 0 :
            st.write('{}컬럼의 최소값'.format(min_max))
            st.dataframe( df.loc[df[min_max] == df[min_max].min(),  ] )
            st.write('{} 컬럼의 최대값'.format(min_max))
            st.dataframe( df.loc[df[min_max] == df[min_max].max(),  ] )            