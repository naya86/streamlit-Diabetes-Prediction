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
from eda_run_app import eda_run_app
import joblib
from deep_ml_app import deep_run_ml_app




def main():
    st.set_page_config(layout='wide', initial_sidebar_state='auto')

    st.title('당뇨병 데이터 및 예측')
    st.subheader('딥러닝을 이용한 당뇨병 예측 앱')

    menu = ['Home', 'EDA', 'Machine Learning']

    choice = st.sidebar.selectbox('Menu', menu)


    if choice == 'EDA' :
        eda_run_app()





    if choice == 'Machine Learning' :
        deep_run_ml_app()








if __name__ == '__main__':
    main()






