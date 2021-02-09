# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import shutil
from os import listdir, mkdir, path
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)
df = pd.DataFrame(columns=['Header', 'Label'])

def addFileToDf(file_path, label):
    global df
    with open(file_path, encoding='us-ascii') as txt:
        try:
            header = txt.readline()
            header = header.replace('Subject: ','',1)
            header = header.replace(' re : ','',1)
            df.loc[-1] = [header, label]
            df.index = df.index + 1
            df = df.sort_index()
        except Exception as e:
            print(e)
            logger.info(f'{file_path} was excluded from csv due to binary encoding')

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def main(input_path, output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    global df
    logger.info('making final data set from raw data')
    
    external_path = 'data/external'
    interim_path = 'data/interim'
    
    if not path.exists(external_path):
        logger.info(f'extracting raw data to {external_path}') 
        mkdir(external_path)
        for tar in listdir(input_path):
            shutil.unpack_archive(f'{input_path}/{tar}', external_path)

    if not path.exists(interim_path):
        logger.info(f'converting external txt files to enron.csv in {interim_path}')
        mkdir(interim_path)
        for enron in listdir(external_path):
            for ham in listdir(f'{external_path}/{enron}/ham'):
                addFileToDf(f'{external_path}/{enron}/ham/{ham}', 0)
            for spam in listdir(f'{external_path}/{enron}/spam'):
                addFileToDf(f'{external_path}/{enron}/spam/{spam}', 1)
        df.to_csv(f'{interim_path}/enron.csv', index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
