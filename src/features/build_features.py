import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from spam_lists import SPAMHAUS_DBL
import spf
import checkdmarc

'''
    Reading the raw data file trec07.csv, and preprocessing the data by getting rid of null values and replacing True,False with 0 and 1 and unknown for some columns
    as well as getting rid of duplicate data
'''
def preprocessing(file_to_read,file_to_write):
    df = pd.read_csv(file_to_read,dtype='unicode')
    cols_for_unknown = ['Return-Path','Message-ID',"From","Reply-To","To","Subject","Date","X-Mailer","Content-Type","Content-Transfer-Encoding"]
    cols_for_zero = ["MIME-Version","X-Priority","X-MSMail-Priority","Status","Content-Length","Lines"]

    for col in cols_for_unknown:
        df[col].fillna(value = "unknown",inplace = True)
    for col in cols_for_zero:
        df[col].fillna("0", inplace = True)
    df.isnull().sum()
    df.drop_duplicates()
    df = df.replace("\n","",regex = True)
    df = df.replace("re :","",regex = True)
    df['new_email'] = df['From'].str.extract(r'([\w\.-]+@[\w\.-]+)')
    df['domain'] = df['new_email'].apply(str).str.split('@').str[1]
    df = df[df['domain'].notna()]
    df.to_csv(file_to_write, index=False)
'''
    Methods for creating new features
'''
def if_special(x):
    special_characters = "]!@#$%^&*()-+?_=,<>/["
    for c in x :
        if c in special_characters:
            return 1
    return 0
def count_num_words(x):
    w = x.split(" ")
    return len(w)

def count_num_cap_words(x):
    w = x.split(" ")
    count = 0
    for i in w:
        if i.isupper():
            count+=1
    return count

def count_num_cap_char(x):
    count = 0
    for i in x:
        if i.isupper():
            count+=1
    return count

def count_digit(x):
    count=0
    for i in x:
        if i.isdigit():
            count+=1
    return count

def count_num_char(x):
    count=0
    for i in x:
        if i.isalpha():
            count+=1
    return count

def count_space(x):
    count=0
    for i in x:
        if i.isspace():
            count+=1
    return count

def count_special(x):
    special_characters = "]!@#$%^&*()-+?_=,<>/["
    return len([c for c in x if c in special_characters])

def singleQuote(x):
    count = 0
    for res in x:
        if "'" in res:
            count+=1
    save = count/2
    return save

def count_num_semiColon(x):
    count = 0
    for i in x:
        if ';' in i:
            count+=1
    return count

def ratio_upperCase_lowerCae(x):

    countUpp =0
    countLow =0

    save = x.split(" ")
    for i in save:
        if i.isupper():
            countUpp+=1
        else:
            countLow+=1

    ratio = countUpp/countLow

    return ratio

def upperCase(x):
    count = 0
    save = x.split(" ")
    for i in save:
        if i.isupper():
            count+=1
    return count

def MaxWordLength(str): 
    strLen = len(str) 
    save = 0; currentLength = 0
      
    for i in range(0, strLen): 
        if (str[i] != ' '): 
            currentLength += 1
        else: 
            save = max(save, currentLength) 
            currentLength = 0

    return max(save, currentLength)
stored_spf = dict()
def check_spf_valid(domain):
    if(domain == ' ' or domain == '' or domain == 'nan'):
        return 0
    if(stored_spf.get(domain)==None):
        try:
            checkdmarc.get_dmarc_record(domain, nameservers=["1.1.1.1"])
            stored_spf[domain] = 1
            return 1
        except:
            stored_spf[domain] = 0
            return 0
    else:
        return stored_spf.get(domain)
stored_val = dict()
def check_blackListed(domain):
    if(domain == ' ' or domain == '' or domain == 'nan'):
        return 0
    if(stored_val.get(domain)==None):
        try:
            if(domain in SPAMHAUS_DBL):
                stored_val[domain] = 1
                return 1
            else:
                stored_val[domain]= 0
                return 0
        except:
            return 0
    else:
        return stored_val.get(domain)
    


def feature_gen(file_to_read,file_to_write):
    df = pd.read_csv(file_to_read,dtype='unicode')
    df['special_characters_exists_subject'] = df['Subject'].apply(if_special)
    df['number_of_words_subject'] = df['Subject'].apply(count_num_words)
    df['number_of_capitalized_words_subject'] = df['Subject'].apply(count_num_cap_words)
    df['number_of_capitalized_characters_subject'] = df['Subject'].apply(count_num_cap_char)
    df['number_of_digits_subject'] = df['Subject'].apply(count_digit)
    df['number_of_characters_subject'] = df['Subject'].apply(count_num_char)
    df['number_of_spaces_subject'] = df['Subject'].apply(count_space)
    df['number_of_special_characters_subject'] = df['Subject'].apply(count_special)
    df['number_of_single_Quotes_subject'] = df['Subject'].apply(singleQuote)
    df['number_of_semiColon_subject'] = df['Subject'].apply(count_num_semiColon)
    df['ratio_of_uppercase/lowercase_words'] = df['Subject'].apply(ratio_upperCase_lowerCae)
    df['Total_number_of_upperCase'] = df['Subject'].apply(upperCase)
    df['Max_word_length_in_subject'] = df['Subject'].apply(MaxWordLength)
    df['spf_valid'] = df.apply(lambda row: check_spf_valid(row['domain']), axis=1)
    df['blackListed'] = df.apply(lambda row: check_blackListed(row['domain']),axis=1)
    '''
    Validating date and subject length
    '''
    df['Date'] = df['Date'].str[:-2]
    #validating date after converting it to datetime
    df['new_date'] = pd.to_datetime(df['Date'],errors="coerce")
    df['validate_date'] = np.where(df['new_date']< datetime.now(), 1, 0)
    df['Subject_length']  = df['Subject'].str.len()
    df.to_csv(file_to_write,index=False)

def main():
    print('Starting Preprocessing')
    preprocessing('data/interim/trec07.csv','data/interim/preprocessed.csv')
    print('Preprocessing Done!')
    print('Starting Feature Gen')
    feature_gen('data/interim/preprocessed.csv','data/interim/data_with_features.csv')
    print('Done Feature Gen')

if __name__ == "__main__":
    main()
