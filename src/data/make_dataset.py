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
columns = ['Return-Path', 'Message-ID', 'From', 'Reply-To', 'To', 'Subject', 'Date', 'X-Mailer', 'Content-Type', 'Content-Length', 'Content-Transfer-Encoding', 'Label']
df = pd.DataFrame(columns=columns)

def addEmailToDf(file_path):
    global df
    
    index_path = 'data/raw/trec07p/full/index'
    mail_id = file_path.split('.')[1]
    header = dict.fromkeys(columns)

    label = 0
    with open(index_path, encoding='us-ascii') as index:
        for i, line in enumerate(index):
            if i == int(mail_id) - 1 and line[0:4] == 'spam':
                label = 1 
    try:
        with open(file_path, encoding='us-ascii') as email:
            for line in email:
                split_line = line.split(':')
                value = split_line[0]
                if value in header:
                    header[value] = split_line[1]           
            df.loc[-1] = [header['Return-Path'], header['Message-ID'], header['From'], header['Reply-To'], header['To'], header['Subject'], header['Date'], header['X-Mailer'], header['Content-Type'], header['Content-Length'], header['Content-Transfer-Encoding'], label]
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
    global df
    logger.info('making final data set from raw data')
    
    interim_path = 'data/interim'
    
    if not path.exists(interim_path):
        logger.info(f'converting external txt files to trec07.csv in {interim_path}')
        mkdir(interim_path)
        for email in listdir(f'{input_path}/trec07p/data'):
            addEmailToDf(f'{input_path}/trec07p/data/{email}')
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
