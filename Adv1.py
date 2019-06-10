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

#Fetch feature from input
X_feature = mydata.iloc[:,1 : 4]

#Fetch target from input
Y_target = mydata.iloc[:,4]
#print(Y_target)

#changing both dataframe input and series target to numpy
X_input = X_feature.values

#split the feature into training set and testing set
X_train = X_input[:150]
X_test = X_input[150:]

#split the target into training and testing set
Y_train =Y_target[:150]
Y_test = Y_target[150:]

#Import LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression
teacher = LinearRegression()


#Train and predict for television
learner = teacher.fit(X_train,Y_train)
print(teacher)
Yp1=learner.predict(X_train)
c1=learner.intercept_
m1=learner.coef_
print("c is{} m is {} and Yp is {}".format(c1,m1,Yp1))
#plot for television 
plt.plot(X_train,Y_train,'r*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.plot(X_train,Yp1,'g*')
plt.xlabel("input")
plt.ylabel("Ya")
plt.legend(["Ya","Yp"])
plt.show()

#Calculate error
Error = mean_squared_error(Yp1,Y_train)
E=np.sqrt(Error)
print("error is:" ,np.sqrt(Error))
