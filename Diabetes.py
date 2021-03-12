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




def main():
    st.set_page_config(layout='wide', initial_sidebar_state='auto')

    st.title('당뇨병 데이터 및 예측')

    menu = ['Home', 'EDA', 'Machine Learning']

    choice = st.sidebar.selectbox('Menu', menu)







    if choice == 'Machine Learning' :
        deep_run_ml_app()








if __name__ == '__main__':
    main()






