# This script loads the raw reddit suicide detection dataset, 
# applies basic cleaning 
# and saves a filtered version to data/processed/

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'suicide_detection.csv'
OUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'posts_clean.csv'

def main():

    #Load raw data

    print(f'Loading data from {RAW_PATH}')
    df = pd.read_csv(RAW_PATH)

    #Basic info of the dataset

    print('\n Raw Dataset Info')
    print(f'Shape: {df.shape}')

    print('\n Class Distribution:')
    print(df['class'].value_counts())

    print('\n Missing values per column')
    print(df.isnull().sum())

    #Dropping NaN and empty rows and index column

    print('\nDropping rows with missing or empty text')
    before = len(df)
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    print(f'\nRemoved {before - len(df)} rows -> {len(df)} remaining')
    df = df.drop(columns=['Unnamed: 0'])
    print('\nIndex column -Unnamed- is removed')

    #Removing duplicates
    print('\nRemoving duplicate posts')
    before = len(df)
    df = df.drop_duplicates(subset=['text'])
    print(f'\nRemoved {before - len(df)} duplicates -> {len(df)} remaining')

    #Stripping whitespaces

    print('\nStripping whitespace in posts')
    df['text'] = df['text'].str.strip()

    #create word_count column (to filter out short posts and use in visualisation)

    print('\nComputing word counts')
    df['word_count'] = df['text'].str.split().str.len()

    #Filtering the posts with fewer than 10 words

    print('\nFiltering posts with fewer than 10 words')
    before = len(df)
    df = df[df['word_count'] >= 10]
    print(f' Removed {before - len(df)} short posts -> {len(df)} remaining')

    #Reset index
    
    df = df.reset_index(drop=True)

    #Print updated shape and class distribution

    print('/n Cleaned Dataset Info')
    print(f'Shape: {df.shape}')
    print('/n Class Distribution')
    print(df['class'].value_counts())

    #Save it to processed folder

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f'/n Saving cleaned data to {OUT_PATH}')
    df.to_csv(OUT_PATH, index=False)
    print('\nDONE')

if __name__ == '__main__':
    main()