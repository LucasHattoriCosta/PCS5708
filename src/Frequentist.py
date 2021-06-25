import pandas as pd
from .utils import *

class TransitionMatrix():
    def __init__(self, df=None, val_df=None):
        self.name = 'Transition Matrix generated purely on frequentist approach'
        if df is None or val_df is None:
            self.df, self.val_df = read_data()
        else:
            self.df, self.val_df = df, val_df
        self.df['rating'] = self.df['n_rating']
        self.compute_transition_matrix()

    def make_prediction(self,prevRating):
        return self.matrix.loc[prevRating]

    def compute_val_score(self):
        val = self.val_df
        yPred = []
        yTrue = []
        for _, row in tqdm(val.iterrows()):
            try:
                nextRating = val.loc[(val['year']==row['year']+1) & (val['Symbol']==row['Symbol']), 'n_rating'].iloc[0]
            except IndexError:
                continue
            probNextRating = self.make_prediction(row['Rating']).values
            yPred.append(list(probNextRating))
            yTrue.append(nextRating)
        return np.array(yPred),np.array(yTrue)


    def compute_transition_matrix(self, y_final: int=2017) -> pd.DataFrame:
        '''
        Computa a matrix de transicao dos ratings utilizando os dados disponiveis em input_df para o periodo ate y_final.
        '''
        df = self.df.copy()
        df = df.sort_values(['year'], ascending=True)
        years = np.arange(2010, y_final)
        freq_matrix = {prev_rating:{next_rating:0 for next_rating in range(7)} for prev_rating in range(7)}
        total_cases_matrix = {prev_rating:0 for prev_rating in range(7)}    
        
        for y in tqdm(years):
            for t in tickers_in_years(y,df):
                prev_rating, next_rating = df.loc[
                    ( (df['Symbol']==t) & ((df['year']==y) | (df['year']==y+1)) ),
                    'rating'
                ][:2]
                freq_matrix[prev_rating][next_rating] += 1
        
        freq_matrix = pd.DataFrame(freq_matrix)
        prev_total_cases = freq_matrix.sum()
        for rating in range(7):
            freq_matrix[rating] /= prev_total_cases[rating]
        transition_matrix = freq_matrix.T.fillna(0)
        for rating in range(7):
            if transition_matrix.loc[rating].sum() == 0:
                transition_matrix.loc[rating, rating] = 1.0
        
        transition_matrix.columns = RATINGS
        transition_matrix.index = RATINGS
        prev_total_cases.index = RATINGS

        self.matrix = transition_matrix
        self.prev_total_cases = prev_total_cases