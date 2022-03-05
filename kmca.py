from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split

class kmca():
      def __init__(self,df,dev,seed,nclus):
          self.df=df
          self.dev=dev
          self.seed=seed
          self.nclus=nclus
          
      def cal(self):
          X=self.df.iloc[:,1:]
          kmeans = KMeans(n_clusters=self.nclus, random_state=42) 
          kmeans.fit(X)
          ncol=pd.DataFrame(kmeans.labels_, columns=['cluster'])
          dfn=pd.concat([self.df,ncol], axis=1)
          #dfn.to_csv('dfn.csv',index=False)
          m,n=[],[]
          gr=dfn.groupby('cluster').count()
          gr2=gr[gr[self.df.iloc[:,0:1].columns[0]]>1].reset_index()
          for i in gr2['cluster'].unique():
              di=dfn[dfn['cluster']==i]
              ai,bi= train_test_split(di,test_size=0.2, random_state=2)
              ai['Set']='Train'
              bi['Set']='Test'
              ad2=pd.concat([ai,bi],axis=0)
              #ad2=ad2[[dfn.iloc[:,0:1].columns[0],'Set']]
              n.append(ad2)
              ad3=pd.concat(n,axis=0)
          gr=dfn.groupby('cluster').count()
          gr1=gr[gr[dfn.iloc[:,0:1].columns[0]]==1].reset_index()
          for i in gr1['cluster'].unique():
              di2=dfn[dfn['cluster']==i]
              di2['Set']='Train'
              m.append(di2)
              ad4=pd.concat(m,axis=0)
          if gr1.shape[0]>0:
             nd=pd.concat([ad3,ad4],axis=0)
          else:
             nd=ad3

          tr=nd[nd['Set']=='Train']
          tr=tr.drop(['cluster','Set'],axis=1)
          ts=nd[nd['Set']=='Test']
          ts=ts.drop(['cluster','Set'],axis=1)
          return tr,ts
          
