import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import os
from tkinter.filedialog import askopenfilename
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from kmca import kmca
from sequential_selection import stepwise_selection as sq
from loo import loo
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
from rm2 import rm2
from applicability import apdom
import math
import numpy as np
#from WilliamsPlot import williams_plot
from matplotlib import pyplot
from stepwise_selection import stepwise_selection as ss
from sequential_selection import stepwise_selection as sq
from sequential_selection_lda import stepwise_selection as sqlda
from testset_prediction import testset_prediction as tsp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


form = tk.Tk()
form.title("SFS-QSAR")
form.geometry("650x350")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)
tab3 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Data preparation")
tab_parent.add(tab2, text="Regression model development")
tab_parent.add(tab3, text="Classification model development")

initialdir=os.getcwd()

reg=LinearRegression()
model=LinearDiscriminantAnalysis()


def data():
    filename = askopenfilename(initialdir=initialdir,title = "Select file")
    firstEntryTabOne.delete(0, END)
    firstEntryTabOne.insert(0, filename)
    global a_
    a_,b_=os.path.splitext(filename)
    global sfile
    sfile = pd.read_csv(filename)
    #print(sfile)

def datatr():
    global filename1
    filename1 = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree.delete(0, END)
    firstEntryTabThree.insert(0, filename1)
    global c_
    c_,d_=os.path.splitext(filename1)
    global file1
    file1 = pd.read_csv(filename1)
   
    
def datats():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
   

def datatr_cl():
    global filename1_cl
    filename1_cl = askopenfilename(initialdir=initialdir,title = "Select sub-training file")
    firstEntryTabThree_cl.delete(0, END)
    firstEntryTabThree_cl.insert(0, filename1_cl)
    global c_
    c_,d_=os.path.splitext(filename1_cl)
    global file1_cl
    file1_cl = pd.read_csv(filename1_cl)
    
    
def datats_cl():
    global filename2_cl
    filename2_cl = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree_cl.delete(0, END)
    secondEntryTabThree_cl.insert(0, filename2_cl)
    global file2_cl
    file2_cl = pd.read_csv(filename2_cl)
    

   
def variance(X,threshold):
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X

def corr(df):
    lt=[]
    df1=df.iloc[:,1:]
    for i in range(len(df1)):
        x=df1.values[i]
        x = sorted(x)[0:-1]
        lt.append(x)
    flat_list = [item for sublist in lt for item in sublist]
    if len(flat_list)>0:
       return max(flat_list),min(flat_list)
    else:
       messagebox.showinfo('Error','No parameter selected, select other values')

def correlation(X,cthreshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = X.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] > float(cthreshold)) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in X.columns:
                   del X[colname] # deleting the column from the dataset
    return X   

def variance(X,threshold):
    sel = VarianceThreshold(threshold=(threshold* (1 - threshold)))
    sel_var=sel.fit_transform(X)
    X=X[X.columns[sel.get_support(indices=True)]]    
    return X


def pretreat(X,cthreshold,vthreshold):
    X=correlation(X,cthreshold)
    X=variance(X,vthreshold)
    return X

def shuffling(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df


def ROCplot(X,y):
    model.fit(X,y)
    lr_probs =model.predict_proba(X)
    lr_probs = lr_probs[:, 1]
    lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)
    return lr_fpr, lr_tpr

def datadiv(): 
    global mconfig
    #mconfig = open(str(a_)+"_datadivparams.txt","w")
    X=sfile.iloc[:,2:]
    Xm=variance(X,float(fourthEntryTabThreer5c2_1.get()))
    sds=sfile.iloc[:,0:2]
    filem=pd.concat([sds,Xm],axis=1)
    #print(filem.iloc[:,1:2])
    #mconfig.write("Variance cut-off: "+str(fourthEntryTabThreer5c2_1.get())+"\n")  
    if Selection.get()=='Activity sorting':
       perc=int(secondEntryTabOne.get())
       pc=int(100/perc)
       sp=int(secondEntryTabOne_x.get())
       filem=filem.sort_values(filem.iloc[:,1:2].columns[0],ascending=False)
       ts=filem.iloc[(sp-1)::pc, :]
       tr=filem.drop(ts.index.values)
       #mconfig.write("Dataset division technique: Activity sorting"+"\n")  
    elif Selection.get()=='Random':
       perc=secondEntryTabOne.get()
       perc=float(perc)/100
       seed=thirdEntryTabOne.get()
       a,b= train_test_split(filem,test_size=perc, random_state=int(seed))
       tr=pd.DataFrame(a)
       ts=pd.DataFrame(b)
       #mconfig.write("Dataset division technique: Random division"+"\n")
    elif Selection.get()=='KMCA':
       perc=secondEntryTabOne.get()
       perc=float(perc)/100
       nclus=thirdEntryTabOne_x.get()
       #nc=firstEntryTabTwo.get()
       seed=thirdEntryTabOne.get()
       kmc=kmca(filem,perc,int(seed),int(nclus))
       tr,ts=kmc.cal()
       #mconfig.write("Dataset division technique: KMCA"+"\n")
    savename1= str(a_) + '_tr.csv'
    tr.to_csv(savename1,index=False)
    #savename2 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save testset file")
    savename2= str(a_) + '_ts.csv'
    ts.to_csv(savename2,index=False)
    #mconfig.close()

def trainsetfit(X,y,ms):
    corrl=thirdEntryTabThreer3c1.get()
    var=fourthEntryTabThreer5c1.get()
    ti=sixthEntryTabThree.get()
    to=seventhEntryTabThree.get()
    X=pretreat(X,float(corrl),float(var))
    print(X.shape[1])
    if X.shape[1]<ms: 
       messagebox.showinfo('Error','Insufficient number of parameters, change cutoff values')
    else:  
       sl=ss(X,y,float(ti),float(to),int(ms))
       a,b,c,d=sl.fit_()
       return a,b,c,d


def trainsetfit2(X,y):
    cthreshold=thirdEntryTabThreer3c2.get()
    vthreshold=fourthEntryTabThreer5c2.get()
    max_steps=fifthBoxTabThreer6c2.get()
    flot=Criterion.get()
    forw=True
    score=Criterion4.get()
    cvl=fifthBoxTabThreer7c2.get()
    #x_,y_=c_.split('_')
    directory=str(OFNEntry.get())
    
    if not os.path.isdir(directory):
       os.mkdir(directory)
    filex = 'Results.txt'
    file_path = os.path.join(directory, filex)
    filer = open(file_path, "w")
    filer.write(str(file_path)+"\n")
    filer.write(str(directory)+"\n")
    filer.write("Correlation cut-off "+str(cthreshold)+"\n")
    filer.write("Variance cut-off "+str(vthreshold)+"\n")
    filer.write("Maximum steps "+str(max_steps)+"\n")
    filer.write("Floating "+str(flot)+"\n")
    filer.write("Scoring "+str(score)+"\n")
    filer.write("Cross_validation "+str(cvl)+"\n")
    filer.write("% of CV increment "+str(fifthLabelTabThreer9c2.get())+"\n")
       
    lt=[0.0001]
    X=pretreat(X,float(cthreshold),float(vthreshold))
    print(X.shape[1])
    if X.shape[1]<int(max_steps):
       messagebox.showinfo('Error','Insufficient number of parameters, change cutoff values') 
    else:
       sqs=sq(X,y,int(max_steps),forw, flot,score,int(cvl))
       a1,b1=sqs.fit_()
       for i in range(1,len(a1)+1,1):
           sqs=sq(X[a1],y,i,flot,TRUE,score,int(cvl))
           a,b=sqs.fit_()
           cv=loo(X[a],y,file1)
           c,m,l=cv.cal()
           lt.append(c)
           val=(lt[len(lt)-1]-lt[len(lt)-2])
           val2=val/lt[len(lt)-2]*100   
           if val2<float(fifthLabelTabThreer9c2.get()):
              break
    tb=X[a].corr()
    mx,mn=corr(tb)
    tbn='correlation.csv'
    tb.to_csv(os.path.join(directory,tbn))
    #pt.to_csv('pt_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    #dt.to_csv('dt.csv',index=False)
    return a,b,c,m,mx,mn,l,filer,directory
    #print(float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))

def trainsetfit4(X,y,ms):
    cthreshold=thirdEntryTabThreer3c1.get()
    vthreshold=fourthEntryTabThreer5c1.get()
    flot=Criterion_t3.get()
    score=Criterion4_t3.get()
    cvl=fifthBoxTabThreer7c2_t3.get()
    X=pretreat(X,float(cthreshold),float(vthreshold))
    print(X.shape[1])
    if X.shape[1]<ms:
       messagebox.showinfo('Error','Insufficient number of parameters, change cutoff values') 
    else:  
       sqs=sqlda(X,y,int(ms),flot,score,int(cvl))
       a,b,c,d=sqs.fit_()
       return a,b,c,d
    
def writefile1():
    Xtr=file1_cl.iloc[:,2:]
    ytr=file1_cl.iloc[:,1:2]
    ntr=file1_cl.iloc[:,0:1]
    lt,ls=[0],[]
    dct=str(OFNEntry_t3.get())
    if not os.path.isdir(dct):
       os.mkdir(dct)
    filex = 'Results.txt'
    file_path = os.path.join(dct, filex)
    filer = open(file_path, "w")

    if var1.get() and Criterionx.get()==True:
       ms=int(fifthBoxTabThreer6c1.get())
       a1,b1,c1,d1=trainsetfit(Xtr,ytr,ms)
       pc=int(flabel2.get())
       filer.write("Note that it is a Increment based selection result"+"\n")
       #filer.write('Method:FS-LDA'+'\n')
       for i in range(1,len(a1)+1,1):
           a,b,c,d=trainsetfit(Xtr[a1],ytr,i)
           model.fit(Xtr[a],ytr)
           lt.append(b)
           ln=len(lt)
           dv=abs(lt[ln-1]-lt[ln-2])
           val2=dv/lt[len(lt)-2]*100
           ls.append(val2)
           if val2<pc:
              break
       filer.write("Increments :"+str(ls)+"\n")
    elif var1.get() and Criterionx.get()==False:
         ms=int(fifthBoxTabThreer6c1.get())
         filer.write("Note that it is not an increment based selection result"+"\n")
         #filer.write('Method:FS-LDA')
         a,b,c,d=trainsetfit(Xtr,ytr,ms)
    elif var2.get() and Criterionx.get()==True:
         ms=int(fifthBoxTabThreer6c1.get())
         a1,b1,c1,d1=trainsetfit4(Xtr,ytr,ms)
         pc=int(flabel2.get())
         filer.write("Note that it is an increment based selection result"+"\n")
         #filer.write('Method:SFS-LDA')
         for i in range(1,len(a1)+1,1):
             a,b,c,d=trainsetfit4(Xtr,ytr,i)
             model.fit(Xtr[a],ytr)
             lt.append(b)
             ln=len(lt)
             dv=abs(lt[ln-1]-lt[ln-2])
             val2=dv/lt[len(lt)-2]*100
             ls.append(val2)
             if val2<pc:
                break
         filer.write("Increments :"+str(ls)+"\n")
    elif var2.get() and Criterionx.get()==False:
         ms=int(fifthBoxTabThreer6c1.get())
         filer.write("Note that it is not a Increment based selection result"+"\n")
         #filer.write('Method:SFS-LDA')
         a,b,c,d=trainsetfit4(Xtr,ytr,ms)

    filer.write("Sub-training set results "+"\n")
    filer.write("\n")
    filer.write('Total Descriptors: '+str(Xtr.shape[0])+'\n')
    #file3.write("Selected features are:"+str(a)+"\n")
    filer.write("Wilks lambda: "+str(b)+"\n")
    filer.write("Fvalue: "+str(c)+"\n")
    filer.write("pvalue: "+str(d)+"\n")
    model.fit(Xtr[a],ytr)
    filer.write("Selected features :"+str(a)+"\n")
    filer.write("intercept: "+str(model.intercept_)+"\n")
    filer.write("coefficients: "+str(model.coef_)+"\n")
    yprtr=pd.DataFrame(model.predict(Xtr[a]))
    yprtr.columns=['Pred']
    yprtr2=pd.DataFrame(model.predict_proba(Xtr[a]))
    yprtr2.columns=['%Prob(-1)','%Prob(+1)']
    adstr=apdom(Xtr[a],Xtr[a])
    yadstr=adstr.fit()
    dfstr=pd.concat([ntr,Xtr[a],ytr,yprtr,yprtr2,yadstr],axis=1)
    dfstr['Set'] = 'Sub_train'
    tb=Xtr[a].corr()
    tbn='correlation.csv'
    #tb.to_csv(tbn) 
    tb.to_csv(os.path.join(dct,tbn),index=False) 
    mx,mn=corr(tb)
    filer.write('Maxmimum intercorrelation between descriptors: '+str(mx)+"\n")
    filer.write('Minimum intercorrelation between descriptors: '+str(mn)+"\n")
    filer.write("\n")      
    if ytr.columns[0] in file2_cl.columns:
       Xts=file2_cl.iloc[:,2:]
       yts=file2_cl.iloc[:,1:2]
       nts=file2_cl.iloc[:,0:1]
       yprts=pd.DataFrame(model.predict(Xts[a]))
       yprts.columns=['Pred']
       yprts2=pd.DataFrame(model.predict_proba(Xts[a]))
       yprts2.columns=['%Prob(-1)','%Prob(+1)']
       adts=apdom(Xts[a],Xtr[a])
       yadts=adts.fit()
       dfsts=pd.concat([nts,Xts[a],yts,yprts,yprts2,yadts],axis=1)
       dfsts['Set'] = 'Test'
       finda=pd.concat([dfstr,dfsts],axis=0)
       savename4= 'Prediction.csv'
       finda.to_csv(os.path.join(dct,savename4),index=False)
       writefile2(Xtr[a],ytr,model,filer)
       filer.write("Test set results: "+"\n")
       filer.write("\n")
       writefile2(Xts[a],yts,model,filer)
       e,f=ROCplot(Xtr[a],ytr)
       g,h=ROCplot(Xts[a],yts)
       pyplot.figure(figsize=(15,10))
       pyplot.plot(e,f, label='Sub-train', color='blue', marker='.',  linewidth=1, markersize=10)
       pyplot.plot(g,h, label='Test', color='red', marker='.', linewidth=1, markersize=10)
       pyplot.ylabel('True postive rate',fontsize=28)
       pyplot.xlabel('False postive rate',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       name4='ROCplot.png'
       pyplot.savefig(os.path.join(dct,name4), dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    else:
       nts=file2_cl.iloc[:,0:1]
       Xts=file2_cl.iloc[:,1:]
       yprts=pd.DataFrame(model.predict(Xts))
       yprts.columns=['Pred']
       yprts2=pd.DataFrame(model.predict_proba(Xvd))
       yprts2.columns=['%Prob(-1)','%Prob(+1)']
       advd=apdom(Xts[a],Xtr[a])
       yadvd=advd.fit()
       dfsvd=pd.concat([nts,Xts,yprts,yprts2,yadvd],axis=1)
       dfsvd['Set'] = 'Screening'
       savename4= 'Prediction.csv'
       dfsvd.to_csv(os.path.join(dct,savename4),index=False)
       e,f=ROCplot(Xtr[a],ytr)
       pyplot.figure(figsize=(15,10))
       pyplot.plot(e,f, label='Train', color='blue', marker='.',  linewidth=1, markersize=10)
       pyplot.ylabel('True postive rate',fontsize=28)
       pyplot.xlabel('False postive rate',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       dct='ROCplot.png'
       pyplot.savefig(os.path.join(dct,name4), dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    if var4.get():
        ls=[]
        nr=int(N1B1_t3.get())
        for i in range(0,nr):
            yr=shuffling(ytr)
            model.fit(Xtr[a],yr)
            ls.append(model.score(Xtr[a],yr))
        averacc=100*np.mean(ls)     
        filer.write('Randomized accuracy after '+str(nr) + ' run: '+str(averacc)+"\n")    
    filer.close()
     
    
def writefilex():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    a,b,c,m,mx,mn,l,filer,dct=trainsetfit2(Xtr,ytr)
    reg.fit(Xtr[a],ytr)
    r2=reg.score(Xtr[a],ytr)
    ypr=pd.DataFrame(reg.predict(Xtr[a]))
    ypr.columns=['Pred']
    rm2tr,drm2tr=rm2(ytr,l).fit()
    #savefile.to_csv('savefile.csv',index=False)
    d=mean_absolute_error(ytr,ypr)
    e=(mean_squared_error(ytr,ypr))**0.5
    adstr=apdom(Xtr[a],Xtr[a])
    yadstr=adstr.fit() 
    df=pd.concat([ntr,Xtr[a],ytr,ypr,l,yadstr],axis=1)
    name1="Train_prediction.csv"
    df.to_csv(os.path.join(dct,name1),index=False)
    
    #filer = open(str(c_)+"_sfslda.txt","w")
    
    filer.write("Sub-training set results "+"\n")
    filer.write("\n")
    filer.write("Selected features are:"+str(a)+"\n")
    filer.write("Statistics:"+str(b)+"\n")
    filer.write('Training set results: '+"\n")
    filer.write('Maxmimum intercorrelation between descriptors: '+str(mx)+"\n")
    filer.write('Minimum intercorrelation between descriptors: '+str(mn)+"\n")
    filer.write('MAE: '+str(d)+"\n")
    filer.write('RMSE: '+str(e)+"\n")
    filer.write('Q2LOO: '+str(c)+"\n")
    
    

    if ytr.columns[0] in file2.columns:
       Xts=file2.iloc[:,2:]
       nts=file2.iloc[:,0:1]
       yts=file2.iloc[:,1:2]
       ytspr=pd.DataFrame(reg.predict(Xts[a]))
       ytspr.columns=['Pred']
       rm2ts,drm2ts=rm2(yts,ytspr).fit()
       tsdf=pd.concat([yts,pd.DataFrame(ytspr)],axis=1)
       tsdf.columns=['Active','Predict']
       tsdf['Aver']=m
       tsdf['Aver2']=tsdf['Predict'].mean()
       tsdf['diff']=tsdf['Active']-tsdf['Predict']
       tsdf['diff2']=tsdf['Active']-tsdf['Aver']
       tsdf['diff3']=tsdf['Active']-tsdf['Aver2']
       r2pr=1-((tsdf['diff']**2).sum()/(tsdf['diff2']**2).sum())
       r2pr2=1-((tsdf['diff']**2).sum()/(tsdf['diff3']**2).sum())
       RMSEP=((tsdf['diff']**2).sum()/tsdf.shape[0])**0.5
       adts=apdom(Xts[a],Xtr[a])
       yadts=adts.fit()
       dfts=pd.concat([nts,Xts[a],yts,ytspr,yadts],axis=1)
       name2="Test_prediction.csv"
       dfts.to_csv(os.path.join(dct,name2),index=False)
       #dfts.to_csv(str(c_)+"_sfslda_tspr.csv",index=False)
       filer.write('rm2LOO: '+str(rm2tr)+"\n")
       filer.write('delta rm2LOO: '+str(drm2tr)+"\n")
       filer.write("\n")
       filer.write('Test set results: '+"\n")
       filer.write('Number of observations: '+str(yts.shape[0])+"\n")
       filer.write('Q2F1/R2Pred: '+ str(r2pr)+"\n")
       filer.write('Q2F2: '+ str(r2pr2)+"\n")
       filer.write('rm2test: '+str(rm2ts)+"\n")
       filer.write('delta rm2test: '+str(drm2ts)+"\n")
       filer.write('RMSEP: '+str(RMSEP)+"\n")
       filer.write("\n")
       plt1=pyplot.figure(figsize=(15,10))
       pyplot.scatter(ytr,ypr, label='Train', color='blue')
       pyplot.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'k--', lw=4)
       pyplot.scatter(yts,ytspr, label='Test', color='red')
       pyplot.ylabel('Predicted values',fontsize=28)
       pyplot.xlabel('Observed values',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       #rocn='obs_vspred.png'
       name3="obs_vspred.png"
       plt1.savefig(os.path.join(dct,name3),dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
       #plt1.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      #format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
       plt2=pyplot.figure(figsize=(15,10))
       pyplot.scatter(ytr,l, label='Train(LOO)', color='blue')
       pyplot.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'k--', lw=4)
       pyplot.scatter(yts,ytspr, label='Test', color='red')
       pyplot.ylabel('Predicted values',fontsize=28)
       pyplot.xlabel('Observed values',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       name4="obsloo_vspred.png"
       plt2.savefig(os.path.join(dct,name4),dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(reg.predict(Xts[a]))
        ytspr.columns=['Pred']
        adts=apdom(Xts[a],Xtr[a])
        yadts=adts.fit()
        dfts=pd.concat([nts,Xts[a],ytspr,yadts],axis=1)
        name2="Test_prediction.csv"
        dfts.to_csv(os.path.join(dct,name2),index=False)
        #dfts.to_csv(str(c_)+"_sfslda_scpr.csv",index=False)
    if var3.get():
        ls=[]
        nr=int(N1B1_x.get())
        for i in range(0,nr):
            yr=shuffling(ytr)
            reg.fit(Xtr[a],yr)
            ls.append(reg.score(Xtr[a],yr))
        rr=np.mean(ls)
        #reg.score(Xtr[a],ytr)
        #r2=b.rsquared
        crp2= math.sqrt(r2)*math.sqrt(r2-rr)
        filer.write('Crp2 after '+str(nr) + ' run: '+str(crp2)+"\n")
    
        
def writefile2(X,y,model,filerw):
    ts=tsp(X,y,model)
    a1,a2,a3,a4,a5,a6,a7,a8,a9,a10=ts.fit()
    filerw.write('True Positive: '+str(a1)+"\n")
    filerw.write('True Negative: '+str(a2)+"\n")
    filerw.write('False Positive '+str(a3)+"\n")
    filerw.write('False Negative '+str(a4)+"\n")
    filerw.write('Sensitivity: '+str(a5)+"\n")
    filerw.write('Specificity: '+str(a6)+"\n")
    filerw.write('Accuracy: '+str(a7)+"\n")
    filerw.write('f1_score: '+str(a8)+"\n")
    #filer.write('Recall score: '+str(recall_score(self.y,ypred))
    filerw.write('MCC: '+str(a9)+"\n")
    filerw.write('ROC_AUC: '+str(a10)+"\n")



def enable0():
    secondLabelTabOne_x['state']='normal'
    secondEntryTabOne_x['state']='normal'
    thirdLabelTabOne['state']='disabled'
    thirdEntryTabOne['state']='disabled'
    thirdLabelTabOne_x['state']='disabled'
    thirdEntryTabOne_x['state']='disabled'           

def enable():
    secondLabelTabOne_x['state']='disabled'
    secondEntryTabOne_x['state']='disabled'
    thirdLabelTabOne['state']='normal'
    thirdEntryTabOne['state']='normal'
    thirdLabelTabOne_x['state']='disabled'
    thirdEntryTabOne_x['state']='disabled'   

def enable2():
    
    thirdLabelTabOne['state']='normal'
    thirdEntryTabOne['state']='normal'
    thirdLabelTabOne_x['state']='normal'
    thirdEntryTabOne_x['state']='normal'   

def enable3():
    N1B1_x['state']='normal'

def enable4():
    flabel1['state']='normal'
    flabel2['state']='normal'
    
def enable5():
    flabel1['state']='disabled'
    flabel2['state']='disabled'

def enable3_t3():
    N1B1_t3['state']='normal'
   
firstLabelTabOne = tk.Label(tab1, text="Select Data",font=("Helvetica", 12))
firstLabelTabOne.place(x=90,y=25)
firstEntryTabOne = tk.Entry(tab1,text='',width=50)
firstEntryTabOne.place(x=180,y=28)
b5=tk.Button(tab1,text='Browse', command=data,font=("Helvetica", 10))
b5.place(x=500,y=25)


fourthLabelTabThreer4c2_1=Label(tab1, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2_1.place(x=200,y=75)
fourthEntryTabThreer5c2_1=Entry(tab1)
fourthEntryTabThreer5c2_1.place(x=315,y=75)


secondLabelTabOne_1=Label(tab1, text='Dataset division techniques',font=('Helvetica 12 bold'))
secondLabelTabOne_1.place(x=220,y=115)

Selection = StringVar()
Criterion_sel1 = ttk.Radiobutton(tab1, text='Activity sorting', variable=Selection, value='Activity sorting',command=enable0)
Criterion_sel2 = ttk.Radiobutton(tab1, text='Random Division', variable=Selection, value='Random',command=enable)
Criterion_sel3 = ttk.Radiobutton(tab1, text='KMCA', variable=Selection, value='KMCA',command=enable2)
Criterion_sel1.place(x=100,y=190)
Criterion_sel2.place(x=280,y=190)
Criterion_sel3.place(x=500,y=190)


secondLabelTabOne=Label(tab1, text='%Data-points(validation set)',font=("Helvetica", 12), justify='center')
secondLabelTabOne.place(x=105,y=150)
secondEntryTabOne=Entry(tab1)
secondEntryTabOne.place(x=315,y=155)


secondLabelTabOne_x=Label(tab1, text='Start point',font=("Helvetica", 12), justify='center',state=DISABLED)
secondLabelTabOne_x.place(x=80,y=215)
secondEntryTabOne_x=Entry(tab1, state=DISABLED)
secondEntryTabOne_x.place(x=60,y=240)

thirdLabelTabOne=Label(tab1, text='Seed value',font=("Helvetica", 12), state=DISABLED)
thirdLabelTabOne.place(x=310,y=215)
thirdEntryTabOne=Entry(tab1, state=DISABLED)
thirdEntryTabOne.place(x=290,y=240)


thirdLabelTabOne_x=Label(tab1, text='Number of clusters',font=("Helvetica", 12), state=DISABLED)
thirdLabelTabOne_x.place(x=480,y=215)
thirdEntryTabOne_x=Entry(tab1, state=DISABLED)
thirdEntryTabOne_x.place(x=485,y=240)


b6=tk.Button(tab1, text='Generate train-test sets', bg="orange", command=datadiv,font=("Helvetica", 10))
b6.place(x=280,y=280)

####TAB2##########
firstLabelTabThree = tk.Label(tab2, text="Select training set",font=("Helvetica", 12))
firstLabelTabThree.place(x=95,y=10)
firstEntryTabThree = tk.Entry(tab2, width=40)
firstEntryTabThree.place(x=230,y=13)
b3=tk.Button(tab2,text='Browse', command=datatr,font=("Helvetica", 10))
b3.place(x=480,y=10)

secondLabelTabThree = tk.Label(tab2, text="Select test/screening set",font=("Helvetica", 12))
secondLabelTabThree.place(x=45,y=40)
secondEntryTabThree = tk.Entry(tab2,width=40)
secondEntryTabThree.place(x=230,y=43)
b4=tk.Button(tab2,text='Browse', command=datats,font=("Helvetica", 10))
b4.place(x=480,y=40)

OFN=Label(tab2, text='Type output folder name',font=("Helvetica", 12))
OFN.place(x=115,y=75)
OFNEntry=Entry(tab2)
OFNEntry.place(x=300,y=78)

L1=Label(tab2, text='Stepwise multiple linear regression',font=("Helvetica 12 bold"))
L1.place(x=210,y=100)

thirdLabelTabThreer2c2=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=220,y=130)
thirdEntryTabThreer3c2=Entry(tab2)
thirdEntryTabThreer3c2.place(x=345,y=130)

fourthLabelTabThreer4c2=Label(tab2, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=220,y=155)
fourthEntryTabThreer5c2=Entry(tab2)
fourthEntryTabThreer5c2.place(x=345,y=155)

fifthLabelTabThreer6c2 = Label(tab2, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=30,y=180)
fifthBoxTabThreer6c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer6c2.place(x=150,y=180)

fifthLabelTabThreer8c2 = Label(tab2, text= '% of CV increment',font=("Helvetica", 12))
fifthLabelTabThreer8c2.place(x=200,y=180)
fifthLabelTabThreer9c2=Entry(tab2)
fifthLabelTabThreer9c2.place(x=345,y=180)

var3= IntVar()
N1 = Checkbutton(tab2, text = "Y-randomization",  variable=var3, \
                 font=("Helvetica", 12), command=enable3)
N1.place(x=480, y=130)

N1B1 = Label(tab2, text= 'Number of runs',font=("Helvetica", 12))
N1B1.place(x=490,y=150)
N1B1_x=Entry(tab2,state=DISABLED)
N1B1_x.place(x=490,y=180)

fifthLabelTabThreer7c2 = Label(tab2, text= 'Cross_validation',font=("Helvetica", 12))
fifthLabelTabThreer7c2.place(x=230,y=210)
fifthBoxTabThreer7c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer7c2.place(x=355,y=210)

Criterion_Label = ttk.Label(tab2, text="Floating:",font=("Helvetica", 12))
Criterion = BooleanVar()
Criterion.set(False)
Criterion_Gini = ttk.Radiobutton(tab2, text='True', variable=Criterion, value=True)
Criterion_Entropy = ttk.Radiobutton(tab2, text='False', variable=Criterion, value=False)
Criterion_Label.place(x=230,y=235)
Criterion_Gini.place(x=300,y=235)
Criterion_Entropy.place(x=350,y=235)

Criterion_Label4 = ttk.Label(tab2, text="Scoring:",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion4 = StringVar()
Criterion4.set('r2')
Criterion_acc3 = ttk.Radiobutton(tab2, text='R2', variable=Criterion4, value='r2')
#Criterion_prec3 = ttk.Radiobutton(tab3, text='Precision', variable=Criterion4, value='precision')
Criterion_roc3 = ttk.Radiobutton(tab2, text='NMAE', variable=Criterion4, value='neg_mean_absolute_error')
Criterion_roc4 = ttk.Radiobutton(tab2, text='NMPD', variable=Criterion4, value='neg_mean_poisson_deviance')
Criterion_roc5 = ttk.Radiobutton(tab2, text='NMGD', variable=Criterion4, value='neg_mean_gamma_deviance')
Criterion_Label4.place(x=230,y=260)
Criterion_acc3.place(x=300,y=260)
Criterion_roc3.place(x=370,y=260)
Criterion_roc4.place(x=440,y=260)
Criterion_roc5.place(x=510,y=260)

b2=Button(tab2, text='Generate model', command=writefilex,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=300,y=285)


########################Tab3###################
# === WIDGETS FOR TAB Three
firstLabelTabThree_cl = tk.Label(tab3, text="Select training set",font=("Helvetica", 12))
firstLabelTabThree_cl.place(x=100,y=10)
firstEntryTabThree_cl = tk.Entry(tab3, width=40)
firstEntryTabThree_cl.place(x=230,y=13)
b3_cl=tk.Button(tab3,text='Browse', command=datatr_cl,font=("Helvetica", 10))
b3_cl.place(x=480,y=10)

secondLabelTabThree_cl = tk.Label(tab3, text="Select test set",font=("Helvetica", 12))
secondLabelTabThree_cl.place(x=120,y=40)
secondEntryTabThree_cl = tk.Entry(tab3,width=40)
secondEntryTabThree_cl.place(x=230,y=43)
b4_cl=tk.Button(tab3,text='Browse', command=datats_cl,font=("Helvetica", 10))
b4_cl.place(x=480,y=40)
#thirdLabelTabThreer1c1=Label(tab3, text='FS-LDA model',font=("Helvetica", 12),anchor=W, justify=LEFT).grid(row=3, column=2)

OFN_t3=Label(tab3, text='Type output folder name',font=("Helvetica", 12))
OFN_t3.place(x=115,y=75)
OFNEntry_t3=Entry(tab3)
OFNEntry_t3.place(x=300,y=78)

Criterionx_Label = ttk.Label(tab3, text="Increment based selection:",font=("Helvetica", 12))
Criterionx = BooleanVar()
Criterionx.set(False)
Criterionx_Gini = ttk.Radiobutton(tab3, text='True', variable=Criterionx, value=True,command=enable4)
Criterionx_Entropy = ttk.Radiobutton(tab3, text='False', variable=Criterionx, value=False,command=enable5)
Criterionx_Label.place(x=50,y=120)
Criterionx_Gini.place(x=240,y=120)
Criterionx_Entropy.place(x=290,y=120)


flabel1 = Label(tab3, text= '% lambda decrease',font=("Helvetica", 12),state=DISABLED)
flabel1.place(x=350,y=120)
flabel2=Entry(tab3, width=10,state=DISABLED)
flabel2.place(x=500,y=120)

#thirdLabelTabThreer2c1=Label(tab3, text='Correlation cutoff',font=("Helvetica", 12)).grid(row=4, column=1)
thirdLabelTabThreer2c1=Label(tab3, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c1.place(x=35,y=155)
thirdEntryTabThreer3c1=Entry(tab3,width=10)
thirdEntryTabThreer3c1.place(x=165,y=158)

#fourthLabelTabThreer4c1=Label(tab3, text='Variance cutoff',font=("Helvetica", 12)).grid(row=5, column=1)
fourthLabelTabThreer4c1=Label(tab3, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c1.place(x=250,y=155)
fourthEntryTabThreer5c1=Entry(tab3,width=10)
fourthEntryTabThreer5c1.place(x=365,y=158)

fifthLabelTabThreer6c1 = Label(tab3, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c1.place(x=440,y=155)
fifthBoxTabThreer6c1= Spinbox(tab3, from_=0, to=100, width=5)
fifthBoxTabThreer6c1.place(x=565,y=158)

var1= IntVar()
C1 = Checkbutton(tab3, text = "FS-LDA",  variable=var1, \
                 font=("Helvetica", 12))
C1.place(x=150, y=180)

sixthLabelTabThree=Label(tab3, text= 'p-value to enter',font=("Helvetica", 12))
sixthLabelTabThree.place(x=35,y=210)
sixthEntryTabThree=Entry(tab3,width=10)
sixthEntryTabThree.place(x=165,y=210)

#seventhLabelTabThree=Label(tab3,text='p-value to remove',font=("Helvetica", 12)).grid(row=8,column=1,sticky=E)
seventhLabelTabThree=Label(tab3,text='p-value to remove',font=("Helvetica", 12))
seventhLabelTabThree.place(x=35,y=245)
seventhEntryTabThree=Entry(tab3, width=10)
seventhEntryTabThree.place(x=165,y=245)

b1=Button(tab3, text='Generate model', command=writefile1,bg="orange",font=("Helvetica", 10))
b1.place(x=150, y=285)

#thirdLabelTabThreer1c2=Label(tab3, text='SFS-LDA model',font=("Helvetica", 12),anchor=W, justify=LEFT).grid(row=3, column=4)
var2= IntVar()
C2 = Checkbutton(tab3, text = "SFS-LDA",  variable=var2, \
                 onvalue = 1, offvalue = 0, font=("Helvetica", 12))
#C2.grid(row=3, column=4)
C2.place(x=300,y=180)

fifthLabelTabThreer7c2_t3 = Label(tab3, text= 'Cross_validation',font=("Helvetica", 12))
fifthLabelTabThreer7c2_t3.place(x=280,y=210)
fifthBoxTabThreer7c2_t3= Spinbox(tab3, from_=0, to=100, width=5)
fifthBoxTabThreer7c2_t3.place(x=405,y=210)

Criterion_Label = ttk.Label(tab3, text="Floating:",font=("Helvetica", 12))
Criterion_t3 = BooleanVar()
Criterion_t3.set(False)
Criterion_Gini = ttk.Radiobutton(tab3, text='True', variable=Criterion, value=True)
Criterion_Entropy = ttk.Radiobutton(tab3, text='False', variable=Criterion, value=False)
Criterion_Label.place(x=280,y=235)
Criterion_Gini.place(x=350,y=235)
Criterion_Entropy.place(x=400,y=235)

Criterion_Label4 = ttk.Label(tab3, text="Scoring:",font=("Helvetica", 12),anchor=W, justify=LEFT)
Criterion4_t3 = StringVar()
Criterion4_t3.set('accuracy')
Criterion_acc3 = ttk.Radiobutton(tab3, text='Accuracy', variable=Criterion4, value='accuracy')
#Criterion_prec3 = ttk.Radiobutton(tab3, text='Precision', variable=Criterion4, value='precision')
Criterion_roc3 = ttk.Radiobutton(tab3, text='ROC_AUC', variable=Criterion4, value='roc_auc')
Criterion_Label4.place(x=280,y=260)
Criterion_acc3.place(x=350,y=260)
#Criterion_prec3.grid(column=4, row=10, sticky=(N))
Criterion_roc3.place(x=430,y=260)

b2=Button(tab3, text='Generate model', command=writefile1,bg="orange",font=("Helvetica", 10),anchor=W, justify=LEFT)
b2.place(x=360,y=285)

var4= IntVar()
N1_t3 = Checkbutton(tab3, text = "Y-randomization",  variable=var4, \
                 font=("Helvetica", 12), command=enable3_t3)
N1_t3.place(x=480, y=180)

N1B1_t3 = Label(tab3, text= 'Number of runs',font=("Helvetica", 12))
N1B1_t3.place(x=490,y=200)
N1B1_t3=Entry(tab3,state=DISABLED)
N1B1_t3.place(x=490,y=230)


tab_parent.pack(expand=1, fill='both')

form.mainloop()
