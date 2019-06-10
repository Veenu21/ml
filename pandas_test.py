#Import pandas as pd
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

#collect data for analysis
mydata = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
print(mydata)

#fetch feature from input data
X_bmi = mydata.iloc[:,0:3]

#fetch target from data
Y_target = mydata.iloc[:,3]

#print type of both feature and target
print("type of X_bmi",type(X_bmi))
print("type of Y_target",type(Y_target))

#Make Series for target
Index_name = pd.Series(["Extremely Weak","Weak","Normal","Overweight","Obesity","Extremely Obesity"])

#map data binary encoding
#X_bmi[X_bmi["Gender"]=="Male"]=0
#X_bmi[X_bmi["Gender"]=="Female"]=1
X_bmi["Gender"] = X_bmi["Gender"].map({"Male":0 ,"Female" : 1})
print(X_bmi)

#changing both dataframe input and series target to numpy
X_input = X_bmi.values
Y_target = Y_target.values

#Again print the type of feature and target
print("type of X_input",type(X_input))
print("type of Y_target",type(Y_target))

#split the feature into training set and testing set
X_train = X_input[:400]
X_test = X_input[400:]

#print the length of training and testing set
print("Sample in X_train",len(X_train))
print("Sample in X_test",len(X_test))

#split the target into training and testing set
Y_train =Y_target[:400]
Y_test = Y_target[400:]

#print the length of training and testing set
print("Sample in Y_train",len(Y_train))
print("Sample in Y_test",len(Y_test))

#apply knn classification model
from sklearn.neighbors import KNeighborsClassifier

#train the data
trainer = KNeighborsClassifier(n_neighbors = 5)

#fit the data
learner = trainer.fit(X_train,Y_train)

#test the data
i=learner.predict([[1,150,50]])
print(Index_name[i])
ya = Y_test
yp = learner.predict(X_test)
yalist=list(ya)
yplist=list(yp)
table= pd.DataFrame({"ya":yalist , "yp" : yplist})
print(table)

from sklearn.metrics import accuracy_score

#checking the accuracy
acc=accuracy_score(ya,yp)*100
print("acc is" , acc)

#finding K value
k_range = range(1,21)
my_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train,Y_train)
    Y_pred = knn.predict(X_test)
    my_score.append(accuracy_score(Y_test,Y_pred)*100)
print(my_score)  

#plotting the graph of accuracy
import matplotlib.pyplot as plt 
plt.plot(k_range,my_score,"ro-")
plt.xlabel("the value of k")
plt.ylabel("Accuracy")
plt.show()
