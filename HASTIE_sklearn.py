import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
arr=[]
arry=[]
  
f=open("C:\Hastie10_2.csv","r")
ContaMax=12000;
Conta=0;
for linea in f:
    Conta=Conta+1
    if Conta > ContaMax :break
    lineadelTrain =linea.split(";")
  
 
    linea_x =[""]
    z=0
    for x in lineadelTrain:
   
        z=z+1
        if z==11: break
        if z==1: linea_x[0]=float(lineadelTrain[z])
        else:  linea_x.append(float(lineadelTrain[z]))
  
    arr.append(linea_x)
    
    if float(lineadelTrain[0])==-1.0:
       arry.append(-1.0)
       
    else:
       arry.append(1.0)
   

x=np.array(arr)

y=np.array(arry)
 
df = pd.DataFrame(x)
df['Y'] = y

# Split into training and test set
train, test = train_test_split(df, test_size = 0.2)
X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]
n_train, n_test = len(X_train), len(X_test)
   
Y_predict_train, Y_predict_test = [np.zeros(n_train), np.zeros(n_test)]

lm= GaussianNB()
lm.fit(X_train,Y_train)   
Y_predict_train=lm.predict(X_train)

Y_train_arr=np.array(Y_train)

TotAciertos=0.0
TotFallos=0.0

   
for i in range (len(Y_predict_train)):
       
    if (Y_predict_train[i]==Y_train_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS NAIVE BAYES")  
print("Total Hits TRAIN = " + str(TotAciertos))
print("Total Failures TRAIN = " + str(TotFallos))

  
Y_predict_test=lm.predict(X_test)

Y_test_arr=np.array(Y_test)

TotAciertos=0.0
TotFallos=0.0


   
for i in range (len(Y_predict_test)):
       
    if (Y_predict_test[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
    
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failures TEST = " + str(TotFallos))
###################################################3
# RandomForestClassifier
#################################################
rf= RandomForestClassifier()

rf.fit(X_train,Y_train)   
Y_predict_train=rf.predict(X_train)

TotAciertos=0.0
TotFallos=0.0
   
for i in range (len(Y_predict_train)):
      
    
    if (Y_predict_train[i]==Y_train_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1

print("")  
print("RESULTS RANDOM FOREST")    
print("Total Hits TRAIN = " + str(TotAciertos))
print("Total Failures TRAIN = " + str(TotFallos))

TotAciertos=0.0
TotFallos=0.0

 
Y_predict_test=rf.predict(X_test)

   
for i in range (len(Y_predict_test)):
      
    
    if (Y_predict_test[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
   
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failures TEST = " + str(TotFallos))
###################################################3
# AdaBoostClassifier
#################################################
ab= AdaBoostClassifier()

ab.fit(X_train,Y_train)   
Y_predict_traint=ab.predict(X_train)

TotAciertos=0.0
TotFallos=0.0

for i in range (len(Y_predict_train)):
        
    if (Y_predict_train[i]==Y_train_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS ADABOOST")    
print("Total Hits TRAIN = " + str(TotAciertos))
print("Total Failures TRAIN = " + str(TotFallos))

   
Y_predict_test=ab.predict(X_test)

TotAciertos=0.0
TotFallos=0.0

for i in range (len(Y_predict_test)):
        
    if (Y_predict_test[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
 
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failures TEST = " + str(TotFallos))
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
