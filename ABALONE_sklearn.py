import numpy as np

import time
Inicio1=time.time()
Fin1=time.time()
Fin=Fin1-Inicio1

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

arr=[]
arry=[]

ContDesde=3133
ContaMax=4177

f=open("C:\\abalone.data","r")
Conta=0;
for linea in f:
    Conta=Conta+1
    if Conta < ContDesde: continue
    if Conta > ContaMax :break
    lineadelTrain =linea.split(",")
  
 
    linea_x =[""]
    z=-1
    for x in lineadelTrain:
   
        z=z+1
        if z==0:
            if lineadelTrain[0] == "M":
              ValorTrain=0.0
            else:
                if lineadelTrain[0] == "F":
                   ValorTrain=1.0
                else:
                        if lineadelTrain[0] == "I":
                            ValorTrain=2.0
                        else:
                            print("Raro se cuela un valor de Sexo no considerado" + lineadelTrain[0])
                            ValorTrain=2.0
        if z==8: break
        if z==0: linea_x[0]=ValorTrain
        else:  linea_x.append(float(lineadelTrain[z]))
  
    arr.append(linea_x)
    
    ClaseLeida=float(lineadelTrain[8])
    
    Clase=0.0    
	
    if (ClaseLeida > 10.0): 
        Clase=2.0
    else:
        if (ClaseLeida > 8.0): 
        		Clase=1.0
    arry.append(Clase)


X_test=np.array(arr)
#   print(x)
Y_test_arr=np.array(arry)

arr=[]
arry=[]


f=open("C:\\abalone.data","r")
ContaMax=3133;
Conta=0;
for linea in f:
    Conta=Conta+1
    if Conta > ContaMax :break
    lineadelTrain =linea.split(",")
  
 
    linea_x =[""]
    z=-1
    for x in lineadelTrain:
   
        z=z+1
        if z==0:
            if lineadelTrain[0] == "M":
              ValorTrain=0.0
            else:
                if lineadelTrain[0] == "F":
                   ValorTrain=1.0
                else:
                        if lineadelTrain[0] == "I":
                            ValorTrain=2.0
                        else:
                            print("Raro se cuela un valor de Sexo no considerado" + lineadelTrain[0])
                            ValorTrain=2.0
        if z==8: break
        if z==0: linea_x[0]=ValorTrain
        else:  linea_x.append(float(lineadelTrain[z]))
  
    arr.append(linea_x)
    
    ClaseLeida=float(lineadelTrain[8])
    
    Clase=0.0    
	
    if (ClaseLeida > 10.0): 
        Clase=2.0
    else:
        if (ClaseLeida > 8.0): 
        		Clase=1.0
    arry.append(Clase)


Fin=Fin1-Inicio1
print ("Time in seconds spent in passing from file in disk to array in memory = " + str(Fin))
X_train=np.array(arr)
#   print(x)
Y_train=np.array(arry)
###################################################3
# Naive Bayes Classifier
#################################################


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
