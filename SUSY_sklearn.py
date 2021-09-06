import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

arr=[]
arry=[]
  

f=open("C:\SUSY.csv","r")
ContaMax=400000;
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
   

x=np.array(arr)
#   print(x)
y=np.array(arry)
 
df = pd.DataFrame(x)
df['Y'] = y

# Split into training and test set
train, test = train_test_split(df, test_size = 0.2,random_state=42)
X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]
n_train, n_test = len(X_train), len(X_test)
  
Y_predict, pred_test = [np.zeros(n_train), np.zeros(n_test)]

lm= GaussianNB()


lm.fit(X_train,Y_train)   
Y_predict=lm.predict(X_train)

Y_train_arr=np.array(Y_train)

TotAciertos=0.0
TotFallos=0.0


for i in range (len(Y_predict)):
   
    
    if (Y_predict[i]==Y_train_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS NAIVE BAYES")     
print("Total Hits TRAIN = " + str(TotAciertos))
print("Total failures TRAIN = " + str(TotFallos))

lm.fit(X_test,Y_test)   
Y_predict=lm.predict(X_test)

Y_test_arr=np.array(Y_test)

TotAciertos=0.0
TotFallos=0.0

   
for i in range (len(Y_predict)):
  
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
     
print("Total Hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))
###################################################3
# RandomForestClassifier
#################################################

rf= RandomForestClassifier()

rf.fit(X_test,Y_test)   
Y_predict=rf.predict(X_test)
# print(Y_predict)
#pp=np.array(Y_test)
# print(pp[2])
TotAciertos=0.0
TotFallos=0.0

   
for i in range (len(Y_predict)):
    # print(Y_predict[i])
   
    
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS GRADIENT FOREST")    
print("Total aciertos TEST = " + str(TotAciertos))
print("Total fallos TEST = " + str(TotFallos))
###################################################3
# AdaBoostClassifier
#################################################
ab= AdaBoostClassifier()
ab.fit(X_test,Y_test)   
Y_predict=ab.predict(X_test)

TotAciertos=0.0
TotFallos=0.0
  
for i in range (len(Y_predict)):
   
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS ADABOOST")    
print("Total Hits TEST = " + str(TotAciertos))
print("Total Failuress TEST = " + str(TotFallos))

