import numpy as np
import pandas as pd
import time
Inicio1=time.time()
Fin1=time.time()
Fin=Fin1-Inicio1
from sklearn.model_selection import train_test_split

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegressionClassifier
#https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
arr=[]
arry=[]

f=open("C:\SUSY.csv","r")
ContDesde=4500000
#ContDesde=0
ContaMax=50000000
Conta=0;
for linea in f:
    Conta=Conta+1
    if Conta < ContDesde: continue
    if Conta > ContaMax :break
    lineadelTrain =linea.split(",")
  
 
    linea_x =[""]
    z=0
    for x in lineadelTrain:
   
        z=z+1
        if z==9: break
        if z==1: linea_x[0]=float(lineadelTrain[z])
        else:  linea_x.append(float(lineadelTrain[z]))
  
    arr.append(linea_x)
    
    if float(lineadelTrain[0])==0.0:
       arry.append(-1.0)
       
    else:
       arry.append(1.0)
   

X_test=np.array(arr)
#   print(x)
Y_test_arr=np.array(arry)
#Y_test_arr=np.array(Y_test)

f=open("C:\SUSY.csv","r")
ContaMax=4500000;
Conta=0;
for linea in f:
    Conta=Conta+1
    if Conta > ContaMax :break
    lineadelTrain =linea.split(",")
  
 
    linea_x =[""]
    z=0
    for x in lineadelTrain:
   
        z=z+1
        if z==9: break
        if z==1: linea_x[0]=float(lineadelTrain[z])
        else:  linea_x.append(float(lineadelTrain[z]))
  
    arr.append(linea_x)
    
    if float(lineadelTrain[0])==0.0:
       arry.append(-1.0)
       
    else:
       arry.append(1.0)
   

Fin1=time.time()
Fin=Fin1-Inicio1
print ("Time in seconds spent in passing from file in disk to array in memory = " + str(Fin))
###################################################3
# Naive Bayes Classifier
#################################################

X_train=np.array(arr)
#   print(x)
Y_train=np.array(arry)

lm= GaussianNB()
lm.fit(X_train,Y_train)   
Y_predict=lm.predict(X_test)



TotAciertos=0.0
TotFallos=0.0


   
for i in range (len(Y_predict)):
   
    
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS NAIVE BAYES")     
print("Total hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))

Fin2=time.time()
Fin=Fin2-Fin1
print ("Time in seconds spent in Naive Bayes = " + str(Fin))
###################################################3
# RandomForestClassifier
#################################################

rf= RandomForestClassifier()
rf.fit(X_train,Y_train)   
Y_predict=rf.predict(X_test)

TotAciertos=0.0
TotFallos=0.0

   
for i in range (len(Y_predict)):
   
    
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS RANDOM FOREST")    
print("Total hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))

Fin3=time.time()
Fin=Fin3-Fin2
print ("Time in seconds spent in Random Forest = " + str(Fin))

###################################################3
# AdaBoostClassifier
#################################################
ab= AdaBoostClassifier()
ab.fit(X_train,Y_train)   
Y_predict=ab.predict(X_test)

TotAciertos=0.0
TotFallos=0.0

#print(Y_test)
   
for i in range (len(Y_predict)):
    
    
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS ADABOOST")    
print("Total Hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))
Fin4=time.time()
Fin=Fin4-Fin3
print ("Time in seconds spent in Adaboost Classifier = " + str(Fin))
###################################################3
# GradientBoostClassifier
#################################################
gb= GradientBoostingClassifier()
gb.fit(X_train,Y_train)   
Y_predict=gb.predict(X_test)

TotAciertos=0.0
TotFallos=0.0
  
for i in range (len(Y_predict)):
   
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS GRADIENT BOOST")    
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failuress TEST = " + str(TotFallos))
Fin5=time.time()
Fin=Fin5-Fin4
print ("Time in seconds spent in Gradient Boost = " + str(Fin))
###################################################3
# LogisticRegressionClassifier
#################################################
lg=LogisticRegression()
lg.fit(X_train,Y_train)   
Y_predict=lg.predict(X_test)

TotAciertos=0.0
TotFallos=0.0
  
for i in range (len(Y_predict)):
   
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS LOGISTIC REGRESSION")    
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failuress TEST = " + str(TotFallos))
Fin6=time.time()
Fin=Fin6-Fin5
print ("Time in seconds spent in Logistic Regression = " + str(Fin))
###################################################3
# DecisionTreeClassifier
#################################################
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)   
Y_predict=dt.predict(X_test)

TotAciertos=0.0
TotFallos=0.0
  
for i in range (len(Y_predict)):
   
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS DECISION TREE")    
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failuress TEST = " + str(TotFallos))
Fin7=time.time()
Fin=Fin7-Fin6
print ("Time in seconds spent in Decision Tree = " + str(Fin))
