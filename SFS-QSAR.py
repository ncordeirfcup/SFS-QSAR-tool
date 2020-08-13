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


form = tk.Tk()
form.title("SFS-QSAR")
form.geometry("650x350")

tab_parent = ttk.Notebook(form)

tab1 = ttk.Frame(tab_parent)
tab2 = ttk.Frame(tab_parent)

tab_parent.add(tab1, text="Data preparation")
tab_parent.add(tab2, text="Model development")

initialdir=os.getcwd()

reg=LinearRegression()


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
    global col1
    col1 = list(file1.head(0))
    
def datats():
    global filename2
    filename2 = askopenfilename(initialdir=initialdir,title = "Select test file")
    secondEntryTabThree.delete(0, END)
    secondEntryTabThree.insert(0, filename2)
    global file2
    file2 = pd.read_csv(filename2)
    #global col2
    #col2 = list(file2.head(0))

   
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
    return max(flat_list),min(flat_list)

def shuffling(df, n=1, axis=0):     
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def datadiv(): 
    global mconfig
    mconfig = open(str(a_)+"_datadivparams.txt","w")
    X=sfile.iloc[:,2:]
    Xm=variance(X,float(fourthEntryTabThreer5c2_1.get()))
    sds=sfile.iloc[:,0:2]
    filem=pd.concat([sds,Xm],axis=1)
    #print(filem.iloc[:,1:2])
    mconfig.write("Variance cut-off: "+str(fourthEntryTabThreer5c2_1.get())+"\n")  
    if Selection.get()=='Activity sorting':
       perc=int(secondEntryTabOne.get())
       pc=int(100/perc)
       sp=int(secondEntryTabOne_x.get())
       filem=filem.sort_values(filem.iloc[:,1:2].columns[0],ascending=False)
       ts=filem.iloc[(sp-1)::pc, :]
       tr=filem.drop(ts.index.values)
       mconfig.write("Dataset division technique: Activity sorting"+"\n")  
    elif Selection.get()=='Random':
       perc=secondEntryTabOne.get()
       perc=float(perc)/100
       seed=thirdEntryTabOne.get()
       a,b= train_test_split(filem,test_size=perc, random_state=int(seed))
       tr=pd.DataFrame(a)
       ts=pd.DataFrame(b)
       mconfig.write("Dataset division technique: Random division"+"\n")
    elif Selection.get()=='KMCA':
       perc=secondEntryTabOne.get()
       perc=float(perc)/100
       nclus=thirdEntryTabOne_x.get()
       #nc=firstEntryTabTwo.get()
       seed=thirdEntryTabOne.get()
       kmc=kmca(filem,perc,int(seed),int(nclus))
       tr,ts=kmc.cal()
       mconfig.write("Dataset division technique: KMCA"+"\n")
    savename1= str(a_) + '_tr.csv'
    tr.to_csv(savename1,index=False)
    #savename2 = filedialog.asksaveasfilename(initialdir=initialdir,title = "Save testset file")
    savename2= str(a_) + '_ts.csv'
    ts.to_csv(savename2,index=False)
    mconfig.close()

def trainsetfit2(X,y):
    cthreshold=thirdEntryTabThreer3c2.get()
    vthreshold=fourthEntryTabThreer5c2.get()
    max_steps=fifthBoxTabThreer6c2.get()
    flot=Criterion.get()
    forw=Criterion3.get()
    score=Criterion4.get()
    cvl=fifthBoxTabThreer7c2.get()
    filer = open(str(c_)+"_results.txt","w")
    filer.write("Correlation cut-off "+str(cthreshold)+"\n")
    filer.write("Variance cut-off "+str(vthreshold)+"\n")
    filer.write("Maximum steps "+str(max_steps)+"\n")
    filer.write("Floating "+str(flot)+"\n")
    filer.write("Scoring "+str(score)+"\n")
    filer.write("Cross_validation "+str(cvl)+"\n")
    filer.write("% of CV increment "+str(fifthLabelTabThreer9c2.get())+"\n")
       
    lt=[0.0001]
    sqs=sq(X,y,float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))
    a1,b1=sqs.fit_()
    for i in range(1,len(a1)+1,1):
        sqs=sq(X[a1],y,float(cthreshold),float(vthreshold),i,flot,forw,score,int(cvl))
        a,b=sqs.fit_()
        cv=loo(X[a],y,file1)
        c,m,l=cv.cal()
        lt.append(c)
        val=(lt[len(lt)-1]-lt[len(lt)-2])
        #print(c)
        #print(val/lt[len(lt)-2]*100)
        val2=val/lt[len(lt)-2]*100   
        if val2<float(fifthLabelTabThreer9c2.get()):
           break
    #print(c)
    tb=X[a].corr()
    mx,mn=corr(tb)
    tbn=str(c_)+'_corr.csv'
    tb.to_csv(tbn)
    #pt.to_csv('pt_train_'+str(cthreshold)+'_'+str(vthreshold)+'.csv')
    #dt.to_csv('dt.csv',index=False)
    return a,b,c,m,mx,mn,l,filer
    #print(float(cthreshold),float(vthreshold),int(max_steps),flot,forw,score,int(cvl))

def writefilex():
    Xtr=file1.iloc[:,2:]
    ytr=file1.iloc[:,1:2]
    ntr=file1.iloc[:,0:1]
    a,b,c,m,mx,mn,l,filer=trainsetfit2(Xtr,ytr)
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
    df.to_csv(str(c_)+"_sfslda_trpr.csv",index=False)
    
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
       dfts.to_csv(str(c_)+"_sfslda_tspr.csv",index=False)
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
       rocn=str(c_)+'_obspred.png'
       plt1.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
       plt2=pyplot.figure(figsize=(15,10))
       pyplot.scatter(ytr,l, label='Train(LOO)', color='blue')
       pyplot.plot([ytr.min(), ytr.max()], [ytr.min(), ytr.max()], 'k--', lw=4)
       pyplot.scatter(yts,ytspr, label='Test', color='red')
       pyplot.ylabel('Predicted values',fontsize=28)
       pyplot.xlabel('Observed values',fontsize=28)
       pyplot.legend(fontsize=18)
       pyplot.tick_params(labelsize=18)
       rocn=str(c_)+'_loopred.png'
       plt2.savefig(rocn, dpi=300, facecolor='w', edgecolor='w',orientation='portrait', papertype=None, \
                      format=None,transparent=False, bbox_inches=None, pad_inches=0.1,frameon=None,metadata=None)
    else:
        Xts=file2.iloc[:,1:]
        nts=file2.iloc[:,0:1]
        ytspr=pd.DataFrame(reg.predict(Xts[a]))
        ytspr.columns=['Pred']
        adts=apdom(Xts[a],Xtr[a])
        yadts=adts.fit()
        dfts=pd.concat([nts,Xts[a],ytspr,yadts],axis=1)
        dfts.to_csv(str(c_)+"_sfslda_scpr.csv",index=False)
    if var3.get():
        ls=[]
        nr=int(N1B1_x.get())
        for i in range(0,nr):
            yr=shuffling(ytr)
            reg.fit(Xtr[a],yr)
            ls.append(reg.score(Xtr[a],yr))
        rr=np.mean(ls)
        reg.score(Xtr[a],ytr)
        #r2=b.rsquared
        crp2= math.sqrt(r2)*math.sqrt(r2-rr)
        filer.write('Crp2 after '+str(nr) + ' run: '+str(crp2)+"\n")
    #wp=williams_plot(Xtr[a],Xts[a],ytr,yts,reg)
    #test_points_in_ad,train_points_in_ad,test_lev_out,h_star,leverage_train,leverage_test,s_residual_train,s_residual_test=wp.plot(toPrint = True,toPlot=True)
    #filer.write("Percetege of train points inside AD: {}%".format(train_points_in_ad))
    #filer.write("Percetege of test points inside AD: {}%".format(test_points_in_ad))
    #filer.write("h*: {}".format(h_star))
        
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

   
firstLabelTabOne = tk.Label(tab1, text="Select Data",font=("Helvetica", 12))
firstLabelTabOne.place(x=90,y=10)
firstEntryTabOne = tk.Entry(tab1,text='',width=50)
firstEntryTabOne.place(x=180,y=13)
b5=tk.Button(tab1,text='Browse', command=data,font=("Helvetica", 10))
b5.place(x=500,y=10)


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

L1=Label(tab2, text='Stepwise multiple linear regression',font=("Helvetica 12 bold"))
L1.place(x=200,y=80)

thirdLabelTabThreer2c2=Label(tab2, text='Correlation cutoff',font=("Helvetica", 12))
thirdLabelTabThreer2c2.place(x=220,y=110)
thirdEntryTabThreer3c2=Entry(tab2)
thirdEntryTabThreer3c2.place(x=345,y=110)

fourthLabelTabThreer4c2=Label(tab2, text='Variance cutoff',font=("Helvetica", 12))
fourthLabelTabThreer4c2.place(x=220,y=135)
fourthEntryTabThreer5c2=Entry(tab2)
fourthEntryTabThreer5c2.place(x=345,y=135)

fifthLabelTabThreer6c2 = Label(tab2, text= 'Maximum steps',font=("Helvetica", 12))
fifthLabelTabThreer6c2.place(x=30,y=160)
fifthBoxTabThreer6c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer6c2.place(x=150,y=160)

fifthLabelTabThreer8c2 = Label(tab2, text= '% of CV increment',font=("Helvetica", 12))
fifthLabelTabThreer8c2.place(x=200,y=160)
fifthLabelTabThreer9c2=Entry(tab2)
fifthLabelTabThreer9c2.place(x=345,y=160)

var3= IntVar()
N1 = Checkbutton(tab2, text = "Y-randomization",  variable=var3, \
                 font=("Helvetica", 12), command=enable3)
N1.place(x=480, y=110)

N1B1 = Label(tab2, text= 'Number of runs',font=("Helvetica", 12))
N1B1.place(x=490,y=130)
N1B1_x=Entry(tab2,state=DISABLED)
N1B1_x.place(x=490,y=160)

fifthLabelTabThreer7c2 = Label(tab2, text= 'Cross_validation',font=("Helvetica", 12))
fifthLabelTabThreer7c2.place(x=230,y=185)
fifthBoxTabThreer7c2= Spinbox(tab2, from_=0, to=100, width=5)
fifthBoxTabThreer7c2.place(x=355,y=185)

Criterion_Label = ttk.Label(tab2, text="Floating:",font=("Helvetica", 12))
Criterion = BooleanVar()
Criterion.set(False)
Criterion_Gini = ttk.Radiobutton(tab2, text='True', variable=Criterion, value=True)
Criterion_Entropy = ttk.Radiobutton(tab2, text='False', variable=Criterion, value=False)
Criterion_Label.place(x=230,y=210)
Criterion_Gini.place(x=300,y=210)
Criterion_Entropy.place(x=350,y=210)

Criterion_Label3 = ttk.Label(tab2, text="Forward:",font=("Helvetica", 12))
Criterion3 = BooleanVar()
Criterion3.set(True)
Criterion_Gini2 = ttk.Radiobutton(tab2, text='True', variable=Criterion3, value=True)
#Criterion_Gini2.pack(column=4, row=9, sticky=(W))
Criterion_Entropy2 = ttk.Radiobutton(tab2, text='False', variable=Criterion3, value=False)
Criterion_Label3.place(x=230,y=235)
Criterion_Gini2.place(x=300,y=235)
Criterion_Entropy2.place(x=350,y=235)


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


tab_parent.pack(expand=1, fill='both')

form.mainloop()
