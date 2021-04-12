'''
Authors: Parthav Desai, Josh Erviho, Daria Patroucheva, Annie Xu
'''

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from spam_lists import SPAMHAUS_DBL
import spf
import checkdmarc

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

'''
    Generating results for Random forest Model
'''
def rfm_model():
    train = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')
    labels = np.array(t['Label'])
    t = t.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)

    # get all the features
    feature_list = list(t.columns)
    features = np.array(t)
    X_train, X_test, y_train, y_test = train_test_split(t, train['Label'], test_size=0.3)
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='1', average='binary')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Metric for Random Forest Model: Precision: {} | Recall: {} | Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
'''
    Generating results for Support Vector Classifier Model
'''
def svc_model():
    train = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')
    labels = np.array(t['Label'])
    t = t.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)

    # get all the features
    feature_list = list(t.columns)
    features = np.array(t)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(t, train['Label'], test_size=0.3)
    from sklearn.svm import SVC
    svclassifier = SVC()
    svclassifer_model = svclassifier.fit(X_train, y_train)
    y_pred = svclassifer_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='1', average='binary')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Metric for Support Vector Classifier: Precision: {} | Recall: {} | Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
'''
Generating results for Gradient Boosted Tree model
'''
def gbt_model():
    # original training data
    train = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')
    labels = np.array(t['Label'])
    t = t.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)

    # get all the features
    feature_list = list(t.columns)
    features = np.array(t)
    X_train, X_test, y_train, y_test = train_test_split(t, train['Label'], test_size=0.3)
    gbt = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.2, max_depth=20, max_features=2)
    gbt_model = gbt.fit(X_train, y_train)
    y_pred = gbt_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='1', average='binary')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Metric for Gradient Boosted Tree: Precision: {} | Recall: {} | Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))

'''
Trains and Saves Recurrent Neural Network Model
'''
def rnn_model():
    
    rnn_filepath = '../../models/rnn_model.pickle'
    data_filepath = '../../data/interim/data_with_features.csv'
    
    print('Preparing training data for RNN model...')

    rnn_df = pd.read_csv(data_filepath, dtype='unicode')
    rnn_df = rnn_df.drop(['Return-Path','Message-ID','From','Reply-To','To','Submitting Host','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines'], axis = 1)

    # split data into testing and training
    test_size = int(len(rnn_df) * 0.3)
    train_data = rnn_df.iloc[:-test_size,:].copy()

    # split training data into labels and features
    features_train = train_data.drop('Label',axis=1).copy()
    label_train = train_data[['Label']].copy()

    # convert df to numpy arrays
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features_train)
    scaled_feature_train = feature_scaler.transform(features_train)
    label_scaler = MinMaxScaler(feature_range=(0, 1))
    label_scaler.fit(label_train)
    scaled_label_train = label_scaler.transform(label_train)
    scaled_label_train = scaled_label_train.reshape(-1)
    scaled_label_train = np.insert(scaled_label_train, 0, 0)
    scaled_label_train = np.delete(scaled_label_train, -1)

    # merge feature and label arrays
    n_input, b_size = 25, 32 
    n_features= features_train.shape[1]
    generator = TimeseriesGenerator(scaled_feature_train, scaled_label_train, length=n_input, batch_size=b_size)
    
    # instantiate sequential model
    print('Instantiating RNN Model...')

    model = Sequential()
    model.add(LSTM(128, input_shape=(n_input, n_features), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    # train model
    print('Training RNN model...')
    model = model.fit_generator(generator,epochs=50)

    # save model
    with open(rnn_filepath, 'wb+') as model_file:
        pickle.dump(model.history, model_file)
    
    print(f'RNN model saved to {rnn_filepath}.')

'''
Trains and Saves Convolutional Neural Network Model
'''
def cnn_model():
    
    cnn_filepath = '../../models/cnn_model.pickle'
    data_filepath = '../../data/interim/data_with_features.csv'
    
    print('Preparing training data for CNN model...')

    cnn_df = pd.read_csv(data_filepath, dtype='unicode')
    cnn_df = cnn_df.drop(['Return-Path','Message-ID','From','Reply-To','To','Submitting Host','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines'], axis = 1)

    # split data into testing and training
    test_size = int(len(rnn_df) * 0.3)
    train_data = cnn_df.iloc[:-test_size,:].copy()
    test_data = cnn_df.iloc[-test_size:,:].copy()
    
    # split training data into labels and features
    features_train = train_data.drop('Label',axis=1).copy()
    label_train = train_data[['Label']].copy()

    # convert df to numpy arrays
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features_train)
    scaled_feature_train = feature_scaler.transform(features_train)
    label_scaler = MinMaxScaler(feature_range=(0, 1))
    label_scaler.fit(label_train)
    scaled_label_train = label_scaler.transform(label_train)
    scaled_label_train = scaled_label_train.reshape(-1)
    scaled_label_train = np.insert(scaled_label_train, 0, 0)
    scaled_label_train = np.delete(scaled_label_train, -1)

    # merge feature and label arrays
    n_input, b_size = 25, 32 
    n_features= features_train.shape[1]
    generator = TimeseriesGenerator(scaled_feature_train, scaled_label_train, length=n_input, batch_size=b_size)
    
    # instantiate sequential model
    print('Instantiating CNN Model...')

    model = Sequential()
    model.add(Conv1D(128, input_shape=(n_input, n_features), kernel_size=5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Conv1D(128,kernel_size=5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Flatten())  # this converts to 1D feature vectors

    model.add(Dense(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    # train model
    print('Training CNN model...')
    model = model.fit_generator(generator,epochs=50)

    # save model
    with open(cnn_filepath, 'wb+') as model_file:
        pickle.dump(model.history, model_file)
    
    print(f'CNN model saved to {cnn_filepath}.')

def main():
    rfm_model()
    svc_model()
    gbt_model()
    rnn_model()
    cnn_model()

if __name__ == "__main__":
    main()




=======
'''
Authors: Parthav Desai, Josh Erviho, Daria Patroucheva, Annie Xu
'''

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from spam_lists import SPAMHAUS_DBL
import spf
import checkdmarc

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

'''
    Generating results for Random forest Model
'''
def rfm_model():
    filepath = '../../models/rfm_model.pickle'
    train = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')
    labels = np.array(t['Label'])
    t = t.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)

    # get all the features
    feature_list = list(t.columns)
    features = np.array(t)
    X_train, X_test, y_train, y_test = train_test_split(t, train['Label'], test_size=0.3)
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
    rf_model = rf.fit(X_train, y_train)
    pickle.dump(rf_model, open(filepath, 'wb')) #saving the model to pickle
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='1', average='binary')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Metric for Random Forest Model: Precision: {} | Recall: {} | Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
'''
    Generating results for Support Vector Classifier Model
'''
def svc_model():
    filepath = '../../models/svc_model.pickle'
    train = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')
    labels = np.array(t['Label'])
    t = t.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)

    # get all the features
    feature_list = list(t.columns)
    features = np.array(t)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(t, train['Label'], test_size=0.3)
    from sklearn.svm import SVC
    svclassifier = SVC()
    svclassifer_model = svclassifier.fit(X_train, y_train)
    pickle.dump(svclassifer_model, open(filepath, 'wb'))#saving the model to pickle
    y_pred = svclassifer_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='1', average='binary')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Metric for Support Vector Classifier: Precision: {} | Recall: {} | Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
'''
Generating results for Gradient Boosted Tree model
'''
def gbt_model():
    filepath = '../../models/gbt_model.pickle'
    # original training data
    train = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../../data/interim/data_with_features.csv',dtype='unicode')
    labels = np.array(t['Label'])
    t = t.drop(['Submitting Host','Label','Return-Path','Message-ID','From','Reply-To','To','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines','new_email','domain','new_date'], axis = 1)

    # get all the features
    feature_list = list(t.columns)
    features = np.array(t)
    X_train, X_test, y_train, y_test = train_test_split(t, train['Label'], test_size=0.3)
    gbt = GradientBoostingClassifier(n_estimators=50, learning_rate = 0.2, max_depth=20, max_features=2)
    gbt_model = gbt.fit(X_train, y_train)
    pickle.dump(gbt_model, open(filepath, 'wb'))#saving the model to pickle
    y_pred = gbt_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='1', average='binary')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print('Metric for Gradient Boosted Tree: Precision: {} | Recall: {} | Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
    

'''
Trains and Saves Recurrent Neural Network Model
'''
def rnn_model():
    
    rnn_filepath = '../../models/rnn_model.pickle'
    data_filepath = '../../data/interim/data_with_features.csv'
    
    print('Preparing training data for RNN model...')

    rnn_df = pd.read_csv(data_filepath, dtype='unicode')
    rnn_df = rnn_df.drop(['new_email', 'domain', 'new_date', 'Return-Path','Message-ID','From','Reply-To','To','Submitting Host','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines'], axis = 1)

    # split data into testing and training
    test_size = int(len(rnn_df) * 0.3)
    train_data = rnn_df.iloc[:-test_size,:].copy()

    # split training data into labels and features
    features_train = train_data.drop('Label',axis=1).copy()
    label_train = train_data[['Label']].copy()

    # convert df to numpy arrays
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features_train)
    scaled_feature_train = feature_scaler.transform(features_train)
    label_scaler = MinMaxScaler(feature_range=(0, 1))
    label_scaler.fit(label_train)
    scaled_label_train = label_scaler.transform(label_train)
    scaled_label_train = scaled_label_train.reshape(-1)
    scaled_label_train = np.insert(scaled_label_train, 0, 0)
    scaled_label_train = np.delete(scaled_label_train, -1)

    # merge feature and label arrays
    n_input, b_size = 25, 32 
    n_features= features_train.shape[1]
    generator = TimeseriesGenerator(scaled_feature_train, scaled_label_train, length=n_input, batch_size=b_size)
    
    # instantiate sequential model
    print('Instantiating RNN Model...')

    model = Sequential()
    model.add(LSTM(128, input_shape=(n_input, n_features), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    # train model
    print('Training RNN model...')
    model = model.fit_generator(generator,epochs=50)

    # save model
    with open(rnn_filepath, 'wb+') as model_file:
        pickle.dump(model.history, model_file)
    
    print(f'RNN model saved to {rnn_filepath}.')

'''
Trains and Saves Convolutional Neural Network Model
'''
'''
def cnn_model():
    
    cnn_filepath = '../../models/cnn_model.pickle'
    data_filepath = '../../data/interim/data_with_features.csv'
    
    print('Preparing training data for CNN model...')

    cnn_df = pd.read_csv(data_filepath, dtype='unicode')
    cnn_df = cnn_df.drop(['Return-Path','Message-ID','From','Reply-To','To','Submitting Host','Subject','Date','X-Mailer','MIME-Version','Content-Type','X-Priority','X-MSMail-Priority','Status','Content-Length','Content-Transfer-Encoding','Lines'], axis = 1)

    # split data into testing and training
    test_size = int(len(rnn_df) * 0.3)
    train_data = rnn_df.iloc[:-test_size,:].copy()
    test_data = cnn_df.iloc[-test_size:,:].copy()
    
    # split training data into labels and features
    features_train = train_data.drop('Label',axis=1).copy()
    label_train = train_data[['Label']].copy()

    # convert df to numpy arrays
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(features_train)
    scaled_feature_train = feature_scaler.transform(features_train)
    label_scaler = MinMaxScaler(feature_range=(0, 1))
    label_scaler.fit(label_train)
    scaled_label_train = label_scaler.transform(label_train)
    scaled_label_train = scaled_label_train.reshape(-1)
    scaled_label_train = np.insert(scaled_label_train, 0, 0)
    scaled_label_train = np.delete(scaled_label_train, -1)

    # merge feature and label arrays
    n_input, b_size = 25, 32 
    n_features= features_train.shape[1]
    generator = TimeseriesGenerator(scaled_feature_train, scaled_label_train, length=n_input, batch_size=b_size)
    
    # instantiate sequential model
    print('Instantiating CNN Model...')

    model = Sequential()
    model.add(Conv1D(128, input_shape=(n_input, n_features), kernel_size=5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Conv1D(128,kernel_size=5))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=1))

    model.add(Flatten())  # this converts to 1D feature vectors

    model.add(Dense(64))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    
    # train model
    print('Training RNN model...')
    model = model.fit_generator(generator,epochs=50)

    # save model
    with open(cnn_filepath, 'wb+') as model_file:
        pickle.dump(model.history, model_file)
    
    print(f'CNN model saved to {cnn_filepath}.')
'''
def main():
    rfm_model()
    svc_model()
    gbt_model()
    #rnn_model()
    #cnn_model()

if __name__ == "__main__":
    main()




>>>>>>> 3673c26e5ac5977dafccd34f291f55341a67bea8
