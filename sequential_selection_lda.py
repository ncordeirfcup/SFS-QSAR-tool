import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.multivariate.manova import MANOVA
#import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
import numpy as np
from testset_prediction import testset_prediction as tsp

class stepwise_selection():
    def __init__(self,X, y,max_steps,flot,score,cvl):
        self.X=X
        self.y=y
        self.max_steps=max_steps
        self.flot=flot
        self.score=score
        self.cvl=cvl
        initial_list=[]
        
    
    
    def fit_linear_reg(self,X,y):
        x=np.ones(X.shape[0])
        x=list(x)
        x=pd.DataFrame(x)
        x.columns=['constant']
        X=pd.concat([X,x],axis=1)
        dp=pd.concat([X,y],axis=1)
        table=MANOVA.from_formula('X.values~ y.values', data=dp).mv_test().results['y.values']['stat']
        Wilks_lambda=table.iloc[0,0]
        F_value=table.iloc[0,3]
        p_value=table.iloc[0,4]
        return Wilks_lambda,F_value,p_value,table
    

    def feature_selection(self,X,y):
        lda=LinearDiscriminantAnalysis(solver='lsqr')
        initial_list=[]
        included=list(initial_list)
        print(X.shape[1])
        sfs1 = sfs(lda,k_features=self.max_steps,floating=self.flot,
               verbose=0,scoring=self.score,cv=self.cvl)
        
        sfs1 = sfs1.fit(X, y)
        a=list(sfs1.k_feature_names_)
        
        return a
        
    def fit_(self):
        lda=LinearDiscriminantAnalysis(solver='lsqr')
        included_features=self.feature_selection(self.X,self.y)
        lda.fit(self.X[included_features],self.y)
        table=self.fit_linear_reg(self.X[included_features],self.y)[3]
        wlambda=self.fit_linear_reg(self.X[included_features],self.y)[0]
        accuracy=lda.score(self.X[included_features],self.y)
        fvalue=self.fit_linear_reg(self.X[included_features],self.y)[1]
        pvalue=self.fit_linear_reg(self.X[included_features],self.y)[2]
        #print('Selected features are: '+str(included_features))
        #print('Wilks lambda: '+str(wlambda))
        #print('F-value: '+str(fvalue))
        #print('p-value: '+str(pvalue))
        #ts=tsp(lda,self.X[included_features],self.y)
        #ts.fit()
        return included_features,wlambda,fvalue,pvalue