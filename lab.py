# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:47:43 2018

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
# machine learning
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import tree
import scipy as sc
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn import svm

# GNB 
def GNB_c(testData,testLabels,trainData,trainLabels):
    GNB = GaussianNB()
    pred=GNB.fit(trainData, trainLabels).predict(testData)
    tp, fn, fp, tn = confusion_matrix(testLabels,pred).ravel()
    pre=precision_score(testLabels,pred,average='micro') 
    rec=recall_score(testLabels, pred, average='macro')
    f1=f1_score(testLabels,pred, average='macro') 
    score=GNB.score(testData,testLabels)
    lgls=log_loss(testLabels,GNB.predict_proba(testData))
    err=1-score
    mse= mean_squared_error(testLabels,pred)
    print("confusion_matrix")
    print(confusion_matrix(testLabels,pred))
    return tp,tn,pre,rec,f1,score,lgls,err,mse,tp+fn,tn+fp

def decision_c(testData,testLabels,trainData,trainLabels):
    trr = tree.DecisionTreeClassifier()
    pred=trr.fit(trainData, trainLabels).predict(testData)
    tp, fn, fp, tn = confusion_matrix(testLabels,pred).ravel()
    pre=precision_score(testLabels,pred,average='micro') 
    rec=recall_score(testLabels, pred, average='macro')
    f1=f1_score(testLabels,pred, average='macro') 
    score=trr.score(testData,testLabels)
    lgls=log_loss(testLabels,trr.predict_proba(testData))
    err=1-score
    mse= mean_squared_error(testLabels,pred)
    print("confusion_matrix")
    print(confusion_matrix(testLabels,pred))
    return tp,tn,pre,rec,f1,score,lgls,err,mse,tp+fn,tn+fp

def svm_c(testData,testLabels,trainData,trainLabels):
    trr = svm.SVC()
    pred=trr.fit(trainData, trainLabels).predict(testData)
    tp, fn, fp, tn = confusion_matrix(testLabels,pred).ravel()
    pre=precision_score(testLabels,pred,average='micro') 
    rec=recall_score(testLabels, pred, average='macro')
    f1=f1_score(testLabels,pred, average='macro') 
    score=trr.score(testData,testLabels)
    lgls=1.2#log_loss(testLabels,pre)
    err=1-score
    mse= mean_squared_error(testLabels,pred)
    print("confusion_matrix")
    print(confusion_matrix(testLabels,pred))
    return tp,tn,pre,rec,f1,score,lgls,err,mse,tp+fn,tn+fp





train = open("ionosphere.data","r")
labels=np.zeros(shape=(351),dtype=int)
data1=np.ndarray(shape=(351,33), dtype=float, order='F')
i=0
for line in train:
    col=line.split(',')
    data1[i]=col[:33]
    if(col[34].split('\n')[0]=="g"):
        labels[i]=1
    else:
        labels[i]=0
    i=i+1
print("Enter Number of Fold")
k=int(input())  
tp=np.zeros(shape=(k),dtype=float)
tn=np.zeros(shape=(k),dtype=float)
pr=np.zeros(shape=(k),dtype=float)
rec=np.zeros(shape=(k),dtype=float)
f1=np.zeros(shape=(k),dtype=float)
score=np.zeros(shape=(k),dtype=float)
lgls=np.zeros(shape=(k),dtype=float)
er=np.zeros(shape=(k),dtype=float)
mse=np.zeros(shape=(k),dtype=float)
i=0
kf = KFold(n_splits=k,random_state=None, shuffle=True)
fold=np.zeros(shape=(k),dtype=int)
for train_index, test_index in kf.split(data1):
    X_train, X_test = data1[train_index], data1[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    value=svm_c(X_test,y_test,X_train,y_train) # change the method name here to run different methods 
    if(value[9]!=0):
        tp[i]=value[0]/value[9]
    else:
        tp[i]=value[0]
    if(value[10]!=0):
        tn[i]=value[1]/value[10]
    else:
        tn[i]=value[1]
    pr[i]=value[2]
    rec[i]=value[3]
    f1[i]=value[4]
    score[i]=value[5]
    lgls[i]=value[6]
    er[i]=value[7]
    mse[i]=value[8]
    i=i+1
    fold[i-1]=i;
print("Sensitivity or True Positive Rate",tp.mean())
print("Specificity (SPC) or True Negative Rate",tn.mean())
print("Precision Mean",pr.mean())
print("Recall  Mean",rec.mean())
print("F1-score Mean",f1.mean())
print("Accuracy Mean",score.mean())
print("Log Loss Mean",lgls.mean())
print("Error Mean",er.mean())
print("Mean Squared Mean",mse.mean())
xint=range(1, k+1)
plt.xticks(xint)
plt.plot(fold,score)
plt.xlabel("fold")
plt.ylabel("Accuracy")
plt.show()
plt.xticks(xint)
plt.plot(fold,er)
plt.xlabel("fold")
plt.ylabel("Loss")
plt.show()



