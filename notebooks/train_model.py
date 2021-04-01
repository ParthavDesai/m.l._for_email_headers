'''
Authors: Parthav Desai, Josh Erviho, Daria Patroucheva, Annie Xi
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

'''
    Generating results for Random forest Model
'''
def rfm_model():
    train = pd.read_csv(r'../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../data/interim/data_with_features.csv',dtype='unicode')
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
    train = pd.read_csv(r'../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../data/interim/data_with_features.csv',dtype='unicode')
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
    train = pd.read_csv(r'../data/interim/data_with_features.csv',dtype='unicode')

    # training data without labels
    t = pd.read_csv(r'../data/interim/data_with_features.csv',dtype='unicode')
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

'''Add your models here Daria and Annie'''

def main():
    rfm_model()
    svc_model()
    gbt_model()

if __name__ == "__main__":
    main()




