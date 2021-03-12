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
    model = joblib.load('data/best_model(random).pkl')
    df = pd.read_csv('data/diabets.csv')
    st.dataframe(df)

    new = np.array([3,88,58,11,54,24,0.26,22])
    new.reshape(1,-1)
    







if __name__=='__main__':
    main()