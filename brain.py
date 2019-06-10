#Import pandas as pd
import pandas as pd

#Import numpy as np
import numpy as np 

#Import pyplot from matplotlib as plt
from matplotlib import pyplot as plt

#Import train_test_split from sklearn.model_selection
from sklearn.model_selection  import  train_test_split

#Import mean_squared_error,r2_score from sklearn.metrics
from sklearn.metrics import mean_squared_error,r2_score


#Collect data for analysis
mydata = pd.read_csv("data_brain_own_model_design.csv")
print(mydata)
print(type(mydata))
print(mydata.info())

#Separate the features from target
X_input = mydata.iloc[:,0:3]
Y_out = mydata.iloc[:,3]
print(Y_out.shape)
print(X_input.head(1))
print(Y_out.head(1))

#Select features and target from input and check correlation
X_input_nu = mydata[['Gender', 'Age Range' , 'Head Size(cm^3)' , 'Brain Weight(grams)']]
print(X_input_nu.head())
print(X_input_nu.corr())

X_input_nu.hist(bins=50)
plt.show()

plt.scatter(X_input_nu.iloc[:,0],Y_out.values,color='blue')
plt.show()

plt.scatter(X_input_nu.iloc[:,1],Y_out.values,color='red')
plt.show()

plt.scatter(X_input_nu.iloc[:,2],Y_out.values,color='red')
plt.show()

#Predicting HEADSIZE , BRAINWEIGHT
XX=X_input_nu.iloc[:,2]
YY=Y_out
print("type of XX IS {} type of YY is {}".format(type(XX),type(YY)))

#Converting series into numpy array
XS=XX.values.reshape(len(XX),1)
YS=YY.values.reshape(len(YY),1)
print("type of XS IS {} type of YS is {}".format(type(XS),type(YS)))

####splitting in train and test
X_train,X_test,Y_train,Y_test = train_test_split(XS,YS,test_size=.30,random_state=100)
print("Sample of X_train {} and X_test {}".format(len(X_train),len(X_test)))
print("Sample of Y_train {} and Y_test {}".format(len(Y_train),len(Y_test)))

###TRAINING
from sklearn.linear_model import LinearRegression
trainer = LinearRegression(fit_intercept=True)
learner = trainer.fit(X_train,Y_train)

##TEST
Ya=Y_test
Yp=learner.predict(X_test)
plt.plot(Y_test,Ya,"r*")
plt.plot(Y_test,Yp,"g*")

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
learner.intercept_
learner.coef_
error = np.sqrt(mean_squared_error(Ya,learner.predict(X_test)))
Rsquare = r2_score(Ya,learner.predict(X_test))
print(error)
print(Rsquare)

