import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import numpy as np
#from testset_prediction import testset_prediction as tsp

class stepwise_selection():
    def __init__(self,X, y,max_steps,forw,flot,score,cvl):
        self.X=X
        self.y=y
        self.max_steps=max_steps
        self.forw=forw
        self.flot=flot
        self.score=score
        self.cvl=cvl
                


    def feature_selection(self,X,y):
        mlr=LinearRegression()
        initial_list=[]
        included=list(initial_list)
        
        sfs1 = sfs(mlr,k_features=self.max_steps,forward=self.forw,floating=self.flot,
               verbose=0,scoring=self.score,cv=self.cvl)
        sfs1 = sfs1.fit(X, y)
        a=list(sfs1.k_feature_names_)
        return a
        
    def fit_(self):
        mlr=LinearRegression()
        a=self.feature_selection(self.X,self.y)
        model=sm.OLS(self.y, sm.add_constant(pd.DataFrame(self.X[a])))
        results=model.fit()
        b=results.summary()
        return a,b