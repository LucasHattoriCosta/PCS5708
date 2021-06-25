'''
Architecture that relies only on microeconomic data, i.e., the ratios.
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import itertools
import pickle

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from pgmpy.inference import BeliefPropagation

from .utils import *

class MicroRatio():
    def __init__(self, name):
        self.name = name
    
    def _compute_n_line(self, micro: tuple, nextRatio: int = 0) -> int:
        n_line= micro[0]*12*3*2+\
                micro[1]*3*2+\
                micro[2]*3+\
                nextRatio
        return int(n_line)
    
    def compute_cpd_matrix(self,df):
        freqs = np.zeros((12*3*3*2,))
        combinations = itertools.product(np.arange(3),np.arange(12),np.arange(2))
        normalizers = {(r,s,sp):0 for (r,s,sp) in combinations}
        probs = np.zeros((12*3*3*2,))
        df = df.sort_values(['year'])


        for y in (df['year'].unique()):
            for t in tickers_in_years(y,df):
                df1 = df.loc[((df['year']==y) | (df['year']==y+1)) & (df['Symbol']==t)]
                prevs = df1.iloc[0]
                ratioIdx = np.where(df1.columns==self.name)[0][0]
                nextRating = df1.iloc[1,ratioIdx]
                aux_var = (prevs[self.name], prevs['sector'], prevs['sp_500'])
                n_line = self._compute_n_line(aux_var, nextRating)
                freqs[n_line] += 1
                normalizers[tuple(int(x) for x in aux_var)] += 1


        for k in normalizers.keys():
            if normalizers[k] == 0:
                # Se nenhum caso foi observado, a probabilidade padrao associada
                # é de 1.0 para o caso de manter o rating.
                n_line= self._compute_n_line(k, k[0])
                probs[n_line] = 1
            else:
                for i in range(3):
                    n_line= self._compute_n_line(k, i)
                    probs[n_line] = freqs[n_line] / normalizers[k]
        cpd_matrix = []
        for i in range(3):
            cpd_row = []
            for tup in normalizers.keys():
                cpd_row.append(probs[self._compute_n_line(tup,i)])
            cpd_matrix.append(cpd_row)
        return cpd_matrix

class NextRating():
    def __init__(self):
        pass
    
    def _compute_n_line(self, micro: tuple, nextRating: int = 0) -> int:
        n_line= micro[0]*7*(3**5)+\
                micro[1]*7*(3**4)+\
                micro[2]*7*(3**3)+\
                micro[3]*7*(3**2)+\
                micro[4]*7*(3**1)+\
                micro[5]*7*(3**0)+\
                nextRating
        return int(n_line)
    
    def compute_cpd_matrix(self,df):
        ## Criar a matrix de probabilidade de NEXT RATING
        freqs = np.zeros(((7**2)*(3**5),))
        combinations = itertools.product(np.arange(7),np.arange(3),np.arange(3),np.arange(3),np.arange(3),np.arange(3))
        normalizers = {tup:0 for tup in combinations}
        probs = np.zeros(((7**2)*(3**5),))
        df = df.sort_values(['year'])


        for y in (df['year'].unique()):
            for t in tickers_in_years(y,df):
                df1 = df.loc[((df['year']==y) | (df['year']==y+1)) & (df['Symbol']==t)]
                prevs = df1.iloc[0]
                nextRating = df1.iloc[1,2]
                aux_var = (prevs['rating'], prevs['liquidity'], prevs['profitability'], prevs['debt'], prevs['cash'],prevs['asset'])
                n_line = self._compute_n_line(aux_var, nextRating)
                freqs[n_line] += 1
                normalizers[tuple(int(x) for x in aux_var)] += 1


        for k in normalizers.keys():
            if normalizers[k] == 0:
                # Se nenhum caso foi observado, a probabilidade padrao associada
                # é de 1.0 para o caso de manter o rating.
                n_line= self._compute_n_line(k, k[0])
                probs[n_line] = 1
            else:
                for i in range(7):
                    n_line= self._compute_n_line(k, i)
                    probs[n_line] = freqs[n_line] / normalizers[k]
        cpd_matrix = []
        for i in range(7):
            cpd_row = []
            for tup in normalizers.keys():
                cpd_row.append(probs[self._compute_n_line(tup,i)])
            cpd_matrix.append(cpd_row)
        return cpd_matrix

class MicroNet():
    def __init__(self):
        self.name = 'Bayesian Network that relies only on microeconomic data to predict credit rating'
        self.chosen_vars = ['currentRatio', 'grossProfitMargin', 'debtEquityRatio', 'assetTurnover', 'operatingCashFlowPerShare']
        self.df, self.val_df = read_data()
        
    def save_model(self, path):
        if not hasattr(self, 'model'):
            print('Buildando com df padrao')
            self.build()
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)

    @staticmethod
    def build_from_pickle(path):
        with open(path,'rb') as file:
            model = pickle.load(file)
        return model

    def build(self, df=None):
        if df is None:
            df = self.df.copy()
        df = enconding_ratios(df,self.chosen_vars)
        df = df[['Symbol','n_sector','n_rating','year']+[f"d_{x}" for x in self.chosen_vars]+['sp500']]
        df.columns = ['Symbol','sector','rating','year','liquidity','profitability','debt','asset','cash','sp_500']

        ## Indenpendent Variables
        prevLiqu = TabularCPD(
            variable='previousLiquidity',variable_card=3,
            values=np.array([df.groupby(['liquidity']).count().iloc[:,0] / df.shape[0]]).T
        )
        prevProf = TabularCPD(
            variable='previousProfitability',variable_card=3,
            values=np.array([df.groupby(['profitability']).count().iloc[:,0] / df.shape[0]]).T
        )
        prevDebt = TabularCPD(
            variable='previousDebt',variable_card=3,
            values=np.array([df.groupby(['debt']).count().iloc[:,0] / df.shape[0]]).T
        )
        prevCash = TabularCPD(
            variable='previousCash',variable_card=3,
            values=np.array([df.groupby(['cash']).count().iloc[:,0] / df.shape[0]]).T
        )
        prevAsset = TabularCPD(
            variable='previousAssetTurnover',variable_card=3,
            values=np.array([df.groupby(['asset']).count().iloc[:,0] / df.shape[0]]).T
        )
        sector = TabularCPD(
            variable='sector',variable_card=12,
            values=np.array([df.groupby(['sector']).count().iloc[:,0] / df.shape[0]]).T
        )
        rating = TabularCPD(
            variable='rating',variable_card=7,
            values=np.array([df.groupby(['rating']).count().iloc[:,0] / df.shape[0]]).T    
        )
        sp500 = TabularCPD(
            variable='sp_500',variable_card=2,
            values=np.array([df.groupby(['sp_500']).count().iloc[:,0] / df.shape[0]]).T    
        ) 

        ## DEPENDENT VARIABLES
        liqu = TabularCPD(
            variable='liquidity',variable_card=3,
            evidence=['previousLiquidity','sector','sp_500'],evidence_card=[3,12,2],
            values=MicroRatio('liquidity').compute_cpd_matrix(df)
        )
        prof = TabularCPD(
            variable='profitability',variable_card=3,
            evidence=['previousProfitability','sector','sp_500'],evidence_card=[3,12,2],
            values=MicroRatio('profitability').compute_cpd_matrix(df)
        )
        debt = TabularCPD(
            variable='debt',variable_card=3,
            evidence=['previousDebt','sector','sp_500'],evidence_card=[3,12,2],
            values=MicroRatio('debt').compute_cpd_matrix(df)
        )
        cash = TabularCPD(
            variable='cash',variable_card=3,
            evidence=['previousCash','sector','sp_500'],evidence_card=[3,12,2],
            values=MicroRatio('cash').compute_cpd_matrix(df)
        )
        asset = TabularCPD(
            variable='asset',variable_card=3,
            evidence=['previousAssetTurnover','sector','sp_500'],evidence_card=[3,12,2],
            values=MicroRatio('asset').compute_cpd_matrix(df)
        )

        ## FINAL VARIABLE
        nextRatingCPD = TabularCPD(
            variable='nextRating',variable_card=7,
            evidence=['rating','liquidity','profitability','debt','cash','asset'],evidence_card=[7,3,3,3,3,3],
            values=NextRating().compute_cpd_matrix(df)
        )

        model = BayesianModel([
            ('sector','liquidity'),
            ('sector','profitability'),
            ('sector','debt'),
            ('sector','cash'),
            ('sector','asset'),
            ('sp_500','liquidity'),
            ('sp_500','profitability'),
            ('sp_500','debt'),
            ('sp_500','cash'),
            ('sp_500','asset'),            
            ('previousLiquidity','liquidity'),    
            ('previousProfitability','profitability'),
            ('previousDebt','debt'),
            ('previousCash','cash'),
            ('previousAssetTurnover','asset'),
            ('liquidity','nextRating'),
            ('profitability','nextRating'),
            ('debt','nextRating'),
            ('cash','nextRating'),
            ('asset','nextRating'),
            ('rating','nextRating') 
        ])

        # Associating the parameters with the model structure.
        model.add_cpds(
            prevLiqu, prevProf, prevDebt, prevCash, prevAsset,
            rating, sector, sp500,
            liqu, prof, debt, cash, asset,
            nextRatingCPD
        )
        assert model.check_model()

        self.model = model

    def create_bayesian_matrix(self):
        if not hasattr(self, 'model'):
            raise ValueError('You need to build the model first! Use the method .build()')
        
        matrix = pd.DataFrame(columns=RATINGS)
        for rat in range(7):
            q = BeliefPropagation(self.model).query(variables=['nextRating'], evidence={'rating':rat})
            matrix.loc[RATINGS[rat]] = q.values

        return matrix    

    def make_prediction(self, evidence: dict):
        q = BeliefPropagation(self.model).query(variables=['nextRating'], evidence=evidence)
        return q.values
    
    def compute_val_score(self):
        if not hasattr(self, 'model'):
            raise ValueError('You need to build the model first! Use the method .build()')
        yPred = []
        yTrue = []
        val = self.val_df.copy()
        
        for _, row in tqdm(val.iterrows()):
            try:
                nextRating = val.loc[(val['year']==row['year']+1) & (val['Symbol']==row['Symbol']), 'n_rating'].iloc[0]
            except IndexError:
                continue
            probNextRating = self.make_prediction(evidence={
                'rating':row['n_rating'],
                'sector':row['n_sector'],
                'sp_500':row['sp500'],
                'previousLiquidity':row['d_currentRatio'],
                'previousProfitability':row['d_grossProfitMargin'],
                'previousDebt':row['d_debtEquityRatio'],
                'previousCash':row['d_assetTurnover'],
                'previousAssetTurnover':row['d_operatingCashFlowPerShare']                              
            })
            yPred.append(list(probNextRating))
            yTrue.append(nextRating)

        return np.array(yPred),np.array(yTrue)

    def LOOCV(self):
        total_error = []
        print(f'Comeca LOOCV - {datetime.now()}')
        for i in tqdm(range(1230)):
            df, ex = select_val_example(df=self.df,chosen_vars=self.chosen_vars,idx=i)
            predIdx, trueIdx = ex.index
            NextRating = ex.loc[trueIdx,'n_rating']
            self.build(df)
            probNextRating = self.make_prediction(evidence={
                'rating':ex.loc[predIdx,'n_rating'],
                'previousLiquidity':ex.loc[predIdx,'d_currentRatio'],
                'previousProfitability':ex.loc[predIdx,'d_grossProfitMargin'],
                'previousDebt':ex.loc[predIdx,'d_debtEquityRatio'],
                'previousCash':ex.loc[predIdx,'d_assetTurnover'],
                'previousAssetTurnover':ex.loc[predIdx,'d_operatingCashFlowPerShare']                              
            }).loc[RATINGS[NextRating],0]

            total_error.append(1-probNextRating)
        
        return total_error

