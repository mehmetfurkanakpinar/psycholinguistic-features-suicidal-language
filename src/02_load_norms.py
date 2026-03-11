#This script loads ,cleans and merges
# two norm databases (SUBTLEXUS AND AoA) and saves it.

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] /'data' / 'raw'
OUT_PATH = Path(__file__).resolve().parents[1] /'data' / 'processed' / 'combined_norms.csv'

def load_subtlex(path):

    #We will only use Lg10WF and Lg10CD columns which gives us the
    #log versions of word frequency and contextual diversity.

    print('\nLoading SUBTLEXUS norms')
    df = pd.read_csv(path)

    #Keep only the columns we need
    df = df[["Word", "Lg10WF", "Lg10CD"]].copy()

    #Standardising the column names
    df = df.rename(columns={
        'Word': 'word',
        'Lg10WF': 'log_word_freq',
        'Lg10CD': 'log_contextual_diversity',
    })

    #Lowercase for consistent merginf with post tokens
    df['word'] = df['word'].str.lower()

    #Since we lowercased the word column, we need to address duplication(e.g.'the' and'The')
    before = len(df)
    df = df.drop_duplicates(subset=['word'], keep='first')
    print(f'SUBTLEXUS: {before} entries -> {len(df)} after deduplication')

    return df

def load_aoa(path):
    # Kuperman et al. (2012) AoA norms
    # Rating.Mean = age (years) at which a word is typically learned

    print('Loading AoA norms')
    df = pd.read_csv(path)

    #Keeping only the columns we need
    df= df[['Word', 'AoA_Kup']].copy()

    #Standardise the column names
    df = df.rename(columns={
        'Word': 'word',
        'AoA_Kup': 'aoa_rating',
    })

    #Lowercase for consistent merging
    df['word'] = df['word'].str.lower()

    #Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset='word', keep='first')
    print(f'AoA norms: {before} entries -> {len(df)} after deduplication')

    return df

def main():

    subtlex = load_subtlex(RAW_DIR / 'SUBTLEXUS.csv')

    aoa = load_aoa(RAW_DIR / 'AoA_51715_words.csv')

    print('\n Merging SUBTLEXUS and AoA norms on word with inner join')
    norms = pd.merge(subtlex, aoa, on='word', how='inner')

    #Lets see how many words survived the merge

    print(f'Words in SUBTLEXUS only: {len(subtlex) - len(norms):>10,}')
    print(f'Words in AoA only: {len(aoa) - len(norms):>10,}')
    print(f'Words in merged norms: {len(norms):>10}')

    #Descriptives for each norm column

    print('\n Descriptive Statistics')
    stats = norms[['log_word_freq', 'log_contextual_diversity', 'aoa_rating']].describe()
    print(stats.loc[['mean', 'std', 'min', 'max']].round(3).to_string())

    #Save the merged norms

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f'\nSaving combined norms to {OUT_PATH}')
    norms.to_csv(OUT_PATH, index=False)
    print('DONE')


if __name__ == '__main__':
    main()















