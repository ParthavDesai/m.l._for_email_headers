# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import shutil
import re
from os import listdir, mkdir, path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)
columns = ['Return-Path', 'Message-ID', 'From', 'Reply-To', 'To', 'Subject', 'Date', 'X-Mailer', 'MIME-Version', 'Content-Type', 'X-Priority', 'X-MSMail-Priority', 'Status', 'Content-Length', 'Content-Transfer-Encoding', 'Lines', 'Label']

def getIndexMap(index_path, data_path):
    
    index = {}

    with open(index_path, encoding='us-ascii') as index_file:
        for i, line in enumerate(index_file):
            type = line[0:4]
            file_path = line.split('/')[2][:-1]
            if type == 'spam':
                index[f'{data_path}{file_path}'] = 1
            else:
                index[f'{data_path}{file_path}'] = 0
     
    return index

def addEmailToDf(file_path, index, df):
    
    header = dict.fromkeys(columns)
    label = index[file_path]

    try:
        with open(file_path, encoding='us-ascii') as email:
            for line in email:
                split_line = line.split(':')
                value = split_line[0]
                if value in header:
                    header[value] = split_line[1]           
            df.loc[-1] = [header['Return-Path'], 
                          header['Message-ID'], 
                          header['From'], 
                          header['Reply-To'], 
                          header['To'], 
                          header['Subject'], 
                          header['Date'], 
                          header['X-Mailer'], 
                          header['MIME-Version'], 
                          header['Content-Type'], 
                          header['X-Priority'], 
                          header['X-MSMail-Priority'], 
                          header['Status'], 
                          header['Content-Length'], 
                          header['Content-Transfer-Encoding'], 
                          header['Lines'], 
                          label]
            df.index = df.index + 1
            df = df.sort_index()
    except UnicodeDecodeError:
        pass

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(input_path, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info('making final data set from raw data')

    index_path = 'data/raw/trec07p/full/index'
    index = getIndexMap(index_path, f'{input_path}/trec07p/data/')
    interim_path = 'data/interim'
    df = pd.DataFrame(columns=columns)

    count = 0
    if not path.exists(interim_path):
        logger.info(f'converting external txt files to trec07.csv in {interim_path}')
        mkdir(interim_path)
        for email in listdir(f'{input_path}/trec07p/data'):
            addEmailToDf(f'{input_path}/trec07p/data/{email}', index, df)
            count += 1
            if count % 1000 == 0:
                logger.info(f'conversion done for {count}/75000 files')
        df.to_csv(f'{interim_path}/trec07.csv', index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
