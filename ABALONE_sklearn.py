import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

#https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

arr=[]
arry=[]
  

f=open("C:\\abalone-1.data","r")
ContaMax=4177;
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

x=np.array(arr)
#   print(x)
y=np.array(arry)
 
df = pd.DataFrame(x)
df['Y'] = y

# Split into training and test set
train, test = train_test_split(df, test_size = 0.2)
X_train, Y_train = train.iloc[:,:-1], train.iloc[:,-1]
X_test, Y_test = test.iloc[:,:-1], test.iloc[:,-1]
n_train, n_test = len(X_train), len(X_test)
   
Y_predict, pred_test = [np.zeros(n_train), np.zeros(n_test)]

lm= GaussianNB()
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
print("")  
print("RESULTS NAIVE BAYES")     
print("Total hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))
###################################################3
# RandomForestClassifier
#################################################
rf= RandomForestClassifier()
rf.fit(X_test,Y_test)   
Y_predict=rf.predict(X_test)

TotAciertos=0.0
TotFallos=0.0


   
for i in range (len(Y_predict)):
    
    
    if (Y_predict[i]==Y_test_arr[i]):
        TotAciertos=TotAciertos+1
    else:
        TotFallos =TotFallos + 1
print("")  
print("RESULTS GRADIENT FOREST")    
print("Total hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))
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
print("Total hits TEST = " + str(TotAciertos))
print("Total failures TEST = " + str(TotFallos))

