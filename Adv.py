#Import pandas as pd
import pandas as pd

#Import numpy as np
import numpy as np 

#Import pyplot from matplotlib as plt
from matplotlib import pyplot as plt

#Import mean_squared_error from sklearn.metrics
from sklearn.metrics import mean_squared_error

#Collect data for analysis
mydata = pd.read_csv("Advertising.csv")
print(mydata)
print(type(mydata))

#Fetch feature from input
X_feature1 = mydata.iloc[:,1]
print(X_feature1)
X_feature2 = mydata.iloc[:,2]
print(X_feature2)
X_feature3 = mydata.iloc[:,3]
print(X_feature3)

#Fetch target from input
Y_target = mydata.iloc[:,4]
print(Y_target)

#changing both dataframe input and series target to numpy
X_input1 = np.array([X_feature1.values])
X_input2 = np.array([X_feature2.values])
X_input3 = np.array([X_feature3.values])
Y_target = np.array([Y_target.values])

#split the feature into training set and testing set
X_train1 = X_input1[:150]
X_test1 = X_input1[150:]
X_train2 = X_input2[:150]
X_test2 = X_input2[150:]
X_train3 = X_input3[:150]
X_test3 = X_input3[150:]

#split the target into training and testing set
Y_train =Y_target[:150]
Y_test = Y_target[150:]

#Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression
teacher = LinearRegression()

#Train and predict for television
learner = teacher.fit(X_train1,Y_train)
print(teacher)
Yp1=learner.predict(X_test1)
c1=learner.intercept_
m1=learner.coef_
print("c is{} m is {} and Yp is {}".format(c1,m1,Yp1))
#plot for television 
plt.plot(X_test1,Y_test,'r*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.plot(X_test1,Yp1,'g*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.legend(["Ya","Yp"])
plt.show()
yalist=list(Y_train)
yplist=list(Yp1)
table= pd.DataFrame({"ya":yalist , "yp" : yplist})
print(table)
#Calculate error
Error = mean_squared_error(Yp1,Y_train)
E=np.sqrt(Error)
print("error is:" ,np.sqrt(Error))
#Y = m1*X_test1+c1+E
#print(Y)


#Train and predict for radio 
learner = teacher.fit(X_train2,Y_train)
print(teacher)
Yp2=learner.predict(X_train2)
c2=learner.intercept_
m2=learner.coef_
print("c is{} m is {} and Yp is {}".format(c2,m2,Yp2))
#plot for radio
plt.plot(X_train2,Y_train,'r*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.plot(X_train2,Yp2,'g*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.legend(["Ya","Yp"])
plt.show()
#Calculate error
Error = mean_squared_error(Yp2,Y_train)
E=np.sqrt(Error)
print(np.sqrt(Error))
#Y = m2*X_test2+c2+E
#print(Y)

#Train and predict for newspaper
learner = teacher.fit(X_train3,Y_train)
print(teacher)
Yp3=learner.predict(X_train3)
c3=learner.intercept_
m3=learner.coef_
print("c is{} m is {} and Yp is {}".format(c3,m3,Yp3))
#plot for newspaper
plt.plot(X_train3,Y_train,'r*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.plot(X_train3,Yp3,'g*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.legend(["Ya","Yp"])
plt.show()
#Calculate error
Error = mean_squared_error(Yp3,Y_train)
E=np.sqrt(Error)
print(np.sqrt(Error))
#Y = m3*X_test3+c3+E
#print(Y)