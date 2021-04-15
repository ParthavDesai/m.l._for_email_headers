import pickle
import pandas as pd
import sys
import numpy as np
from src.features.build_features import preprocessing, feature_gen

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, MaxPooling1D, Bidirectional,LSTM


import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#GBT Imports
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier

#SVC Imports
from collections import Counter
from sklearn.svm import SVC

#RNN Imports
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense, Dropout, Activation, Input, BatchNormalization, MaxPooling1D, Bidirectional,LSTM


#CNN Imports
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPool1D, Flatten , Embedding, GlobalMaxPool1D
from keras.models import Model
from keras.wrappers.scikit_learn import KerasClassifier



def preprocessing(filename):
     try:
        preprocessing(filename,"../../data/interim/preprocessed_pred_data.csv")
        feature_gen('../../data/interim/preprocessed_pred_data.csv','../../data/feature_gen_pred_data.csv')
     except:
        print('Wrong file path provided, please provide the correct path to the csv')
def supervised_models():
    test_dataset = pd.read_csv(r'../../data/feature_gen_pred_data.csv')
    #reading loaded model
    rfm_filepath = '../../models/rfm_model.pkl'
    svc_filepath = '../../models/svc_model.pkl'
    gbt_filepath = '../../models/gbt_model.pkl'
    
    test_dataset.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)
    feature_list = list(test_dataset.columns)
    features = np.array(test_dataset)
    rfm_load_model = pickle.load(open(rfm_filepath, 'rb')) 
    svc_load_model  = pickle.load(open(svc_filepath, 'rb')) 
    gbt_load_model = pickle.load(open(gbt_filepath, 'rb')) 
    

    rfm_prediction = rfm_load_model.predict(features) 
    svc_prediction = svc_load_model.predict(features)
    gbt_prediction = gbt_load_model.predict(features)
    print('rfm:'+rfm_prediction)
    print('gbt:'+gbt_prediction)
    print('gbt:'+gbt_prediction)
    
def rnn_model():
    rnn_filepath = '../../models/rnn_model.pkl'
    cnn_scaler_filepath = '../../models/cnn_scaler.pkl'
    test_dataset = pd.read_csv(r'../../data/feature_gen_pred_data.csv')
    #this part dnt know about
    rnn_df = rnn_df.drop(['new_email', 'domain', 'new_date', 'Return-Path','Message-ID','From','Reply-To','To','Submitting Host','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines'], axis = 1)
    rnn_scaler = pickle.load(open('rnn_scaler.pkl','rb'))
    rnn_load_model = pickle.load(open(rnn_filepath, 'rb'))
    test_scaled = rnn_scaler.transform(test_dataset)
    prediction_value = rnn_load_model.predict(test_scaled)
    print(prediction_value)

def cnn_model():
    cnn_filepath = '../../models/cnn_model.pickle'
    cnn_scaler_filepath = '../../models/cnn_scaler.pkl'
    test_dataset = pd.read_csv(r'../../data/feature_gen_pred_data.csv')
    #not sure about this part
    cnn_df = cnn_df.drop(['Return-Path','Message-ID','From','Reply-To','To','Submitting Host','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines'], axis = 1)

    rnn_scaler = pickle.load(open('cnn_scaler.pkl','rb'))
    rnn_load_model = pickle.load(open(cnn_filepath, 'rb'))
    test_scaled = rnn_scaler.transform(test_dataset)
    prediction_val = rnn_load_model.predict(test_scaled)
    print(prediction_val)


def main():
    #filepath = input("Enter your filepath: ")
    preprocessing(r"c:\Users\Parthav\Desktop\temp_data.csv")
    supervised_models()
    rnn_model()
    cnn_model()

if __name__ == "__main__":
    main()