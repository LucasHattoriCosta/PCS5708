import pandas as pd
import numpy as np
from tqdm import tqdm


### GLOBAL_VARS
RATINGS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
chosen_vars = ['currentRatio', 'grossProfitMargin', 'debtEquityRatio', 'assetTurnover', 'operatingCashFlowPerShare']
def read_and_treat_data(silence=True) -> pd.DataFrame:
    ### DATA TREATMENT
    df = pd.read_csv('./data/corporate_rating.csv')

    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = [x.year for x in df['Date']]

    # Enconding RATINGS and eliminating irrelevant ones
    df['Rating'] = ['CCC' if (x in ['CC','C','D']) else x for x in df['Rating'] ]
    rating_encoder = {rat:i for (i,rat) in enumerate(RATINGS)}
    df['n_rating'] = df['Rating'].map(rating_encoder)

    # Enconding SECTORS
    df['n_sector'] = df['Sector'].map({s:i for (i,s) in enumerate(sorted(df['Sector'].unique()))})

    # Counting, treating or removing tickers with more than one ocurrence in the same year.
    j=0
    df = df.sort_values(['Date'])
    for y in sorted(df['year'].unique()):
        df1 = df.loc[df['year']==y].copy()
        i=0
        for t in df1.Symbol.unique():
            if df.loc[(df['year']==y) & (df['Symbol']==t)].shape[0] > 1:
                i+=1
                if df.loc[(df['year']==y-1) & (df['Symbol']==t)].shape[0] == 0:
                    df.loc[(df['year']==y) & (df['Symbol']==t), 'year'] = [y-1,y]
                elif df.loc[(df['year']==y+1) & (df['Symbol']==t)].shape[0] == 0:
                    df.loc[(df['year']==y) & (df['Symbol']==t), 'year'] = [y,y+1]
                else:
                    j+=1
                    idx1, idx2 = df.loc[(df['year']==y) & (df['Symbol']==t)].index
                    df.drop([idx2], inplace=True)
        if i > 0 and not silence:
            print(f'Number of repeated ticker in same year for year {y}: {i}')
    if not silence:
        print(f'Number of dropped rows: {j}')

    # Adding S&P500 data
    sp500_enconding = {
        2009: 0,
        2010: 0,
        2011: 1,
        2012: 1,
        2013: 1,
        2014: 1,
        2015: 0,
        2016: 0,
        2017: 1
    } # desenvolvido no notebook SP500
    df['sp500'] = df['year'].map(sp500_enconding)

    # Counting ocurrences that will be useful in the work
    total = 0
    TICKERS_YEARS = {y:[] for y in df['year'].unique()}
    for y in sorted(df['year'].unique()):
        df1 = df.loc[df['year']==y].copy()
        i=0
        for t in df1.Symbol.unique():
            if not df.loc[(df['year']==y+1) & (df['Symbol']==t)].empty:
                i+=1
                TICKERS_YEARS[y].append(t)
        if i>0 and not silence:
            print(f'Numero de tickers sendo avaliados seguidamente entre os anos {y} e {y+1}: {i}')
        total+=i
    if not silence:
        print(f'Numero total de tickers sendo avaliados seguidamente: {total}')
    
    return df

def read_data():
    df = pd.read_csv('./data/training_dataset.csv')
    val_df = pd.read_csv('./data/validation_dataset.csv')
    return df, val_df

def tickers_in_years(y,df):
    tickers = []
    df1 = df.loc[df['year']==y].copy()
    for t in df1['Symbol'].unique():
        if not df.loc[(df['year']==y+1) & (df['Symbol']==t)].empty:
            tickers.append(t)
    return tickers


def enconding_ratios(input_df: pd.DataFrame, chosen_vars: list) -> pd.DataFrame:
    df = input_df.copy()
    numerical_df = df[['Sector']+chosen_vars]
    q30 = numerical_df.groupby(['Sector']).quantile(0.30)
    q70 = numerical_df.groupby(['Sector']).quantile(0.70)

    for sec in (numerical_df['Sector'].unique()):
        for feat in chosen_vars:
            df.loc[
                (df['Sector']==sec) & (df[feat]<=q30.loc[sec,feat]),
                f'd_{feat}'] = 0
            df.loc[
                (df['Sector']==sec) & (df[feat]>=q70.loc[sec,feat]),
                f'd_{feat}'] = 2
            df.loc[
                (df['Sector']==sec) & (df[feat]<q70.loc[sec,feat]) & (df[feat]>q30.loc[sec,feat]),
                f'd_{feat}'] = 1
    return df

def select_val_example(df: pd.DataFrame, chosen_vars: list , idx: int = None) -> pd.DataFrame:
    if idx is None:
        idx = np.random.randint(1230) # 1230 numero de ticker sendo avaliados seguidamente
    
    i = 0
    for y in np.arange(2009,2018):
        ticks = tickers_in_years(y,df)
        i += len(ticks)
        if ticks and i >= idx:
            diff = i - idx
            ex_id = (y,ticks[-diff])
            break
    df_ratios_encoded = enconding_ratios(df, chosen_vars)

    ex = df_ratios_encoded.loc[
        (df['Symbol']==ex_id[1]) &\
        ((df['year']==ex_id[0]) | (df['year']==ex_id[0]+1))
        ]
    df.drop(ex.index, inplace=True)

    return df, ex


