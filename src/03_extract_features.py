# This script looks up psycholinguistic norms for every word 
# for each Reddit post and compute post-level aggregate features:

# - mean_log_freq: mean log10 word frequency (SUBTLEXUS Lg10WF)
# - mean_log_cd: mean log10 contextual diversity (SUBTLEXUS Lg10CD)
# - mean_aoa: mean age of acquisition (Kuperman AoA_Kup)
# - norm_coverage: proportion of post tokens found in the norms

# Output: data/processed/posts_with_features.csv

import string
import numpy as np
import pandas as pd 
from tqdm import tqdm
from pathlib import Path

POSTS_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'posts_clean.csv'
NORMS_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'combined_norms.csv'
OUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'processed' / 'posts_with_features.csv'

punct = string.punctuation

def build_norms_dict(norms_df: pd.DataFrame) -> dict:

    norms_dict = {}

    for row in norms_df.itertuples(index=False):
        norms_dict[row.word] = {
            'log_word_freq': row.log_word_freq,
            'log_contextual_diversity': row.log_contextual_diversity,
            'aoa_rating': row.aoa_rating,
        }
    return norms_dict

def extract_post_features(text:str, norms_dict:dict) -> dict:

    #Extract psycholinguistic features for a single post.

    #Tokenises by lowercasing and splitting on whitespace, then strips
    #leading/trailing punctuation. Contractions (e.g. "don't") are left
    #intact as their lowercase forms exist in the norms.

    #norm_coverage = matched tokens / total tokens. Returns NaN (not 0)
    #when no tokens matched, so downstream code can distinguish missing
    #data from a genuine zero.

    #Lowercasing
    text = text.lower()

    #Tokenising
    tokens = text.split()

    #Stripping punctuation
    tokens = [t.strip(punct) for t in tokens]

    #Remove empty strings that occured after stripping
    tokens = [t for t in tokens if t]
    total_tokens = len(tokens)

    #If somehow there is till empty posts after tokenisation, this returns them NaN
    if total_tokens == 0:
        return{
            'mean_log_freq': np.nan,
            'mean_log_cd': np.nan,
            'mean_aoa': np.nan,
            'norm_coverage': 0.0,
        }
    
    #Collecting norm values for every token in the lookup dict
    freq_vals = []
    cd_vals = []
    aoa_vals = []
    matched = 0

    for token in tokens:
        entry = norms_dict.get(token)
        if entry is not None:
            freq_vals.append(entry['log_word_freq'])
            cd_vals.append(entry['log_contextual_diversity'])
            aoa_vals.append(entry['aoa_rating'])
            matched += 1

    #compute means and return nan if no tokens were matched at all
    mean_log_freq = np.mean(freq_vals) if freq_vals else np.nan
    mean_log_cd = np.mean(cd_vals) if cd_vals else np.nan
    mean_aoa = np.mean(aoa_vals) if aoa_vals else np.nan

    #proportion of the tokens covered by the norms
    norm_coverage = matched/total_tokens

    return {
        "mean_log_freq": mean_log_freq,
        "mean_log_cd":   mean_log_cd,
        "mean_aoa":      mean_aoa,
        "norm_coverage": norm_coverage,
        }


def main():

    #Load cleaned posts
    print(f'Loading posts from {POSTS_PATH}')
    posts = pd.read_csv(POSTS_PATH)
    print(f' {len(posts)} posts loaded')

    #Load combined norms
    print(f'Loading norms from {NORMS_PATH}')
    norms_df = pd.read_csv(NORMS_PATH)
    print(f'{len(norms_df)} words in norms')

    #Build lookup dictionary
    print('Building norms lookup dictionary')
    norms_dict = build_norms_dict(norms_df)
    print(f' Dictionary contains {len(norms_dict):,} entries')

    #Because its a big dataset we apply extract_post_features function 
    # row by row
    print('\nExtracting features...')
    results = []
    for text in tqdm(posts['text'], total = len(posts), unit='post'):
        results.append(extract_post_features(text, norms_dict))

    #Adding the feature columns to our dataset
    features_df = pd.DataFrame(results)
    posts = pd.concat([posts, features_df], axis=1)

    #Statistics for the new feature columns
    feature_cols = ['mean_log_freq', 'mean_log_cd', 'mean_aoa', 'norm_coverage']
    print('\n -Feature Summary Statistics-')
    stats = posts[feature_cols].describe().loc[['mean', 'std', 'min', 'max']]
    print(stats.round(3).to_string())

    #Mean norm coverage across all posts
    mean_cov = posts['norm_coverage'].mean()
    print(f'\nMean norm coverage across all posts: {mean_cov:.1%}')

    #Dropping the rows where all 3 norm features are nan (because of possible slang words)
    norm_feature_cols = ['mean_log_freq', 'mean_log_cd', 'mean_aoa']
    before = len(posts)
    posts = posts.dropna(subset=norm_feature_cols, how='all')
    dropped = before - len(posts)
    print(f'\nDropped {dropped} posts with zero norm matches -> {len(posts):,} remaining')

    #Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f'Saving to {OUT_PATH}')
    posts.to_csv(OUT_PATH, index=False)
    print('DONE')

if __name__ == '__main__':
    main()
