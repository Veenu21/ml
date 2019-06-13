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

#Import seaborn as sb
import seaborn as sb

#Collect data for analysis
mydata = pd.read_csv("pima-indians-diabetes.csv")
print(mydata)
#print(type(mydata))
#print(mydata.info())
#print(mydata.dtypes)
#print(mydata.keys)
X_data = mydata[["Number_pregnant","Glucose_concentration","Blood_pressure" ,"Triceps","Insulin","BMI","Pedigree","Age"]]
print(X_data.head())
Y_data = mydata["Class"]
print("type of X_data is{} and type of Y_data is {} ".format(type(X_data),type(Y_data)))

#converting pandas and series into numpy
XA=np.asanyarray(X_data)
YA=np.asanyarray(Y_data)
print("type of X_data is{} and type of Y_data is {} ".format(type(XA),type(YA)))

#Down feature scaling(preprocessing)
from sklearn import preprocessing
XA = preprocessing.StandardScaler().fit(XA).transform(XA)
#XA[:]
XA_mean=XA.mean()
XA_std=XA.std()
print("mean of XA {} and std of XA {}".format(abs(round(XA_mean)),XA_std))

###splitting in train and test
X_train,X_test,Y_train,Y_test = train_test_split(XA,YA,test_size=.30,random_state=101)
print("Sample of X_train {} and X_test {}".format(len(X_train),len(X_test)))
print("Sample of Y_train {} and Y_test {}".format(len(Y_train),len(Y_test)))

##Trainer
from sklearn.linear_model import LogisticRegression
trainer = LogisticRegression(solver='liblinear')
learner = trainer.fit(X_train,Y_train)
#X_test ......Y_test
Yp = learner.predict(X_test)
Ya = Y_test
Ypprob = learner.predict_proba(X_test)
print(Ypprob)
#probability of y for given x
PY1X=Ypprob[:,0]
P1_Y0=Ypprob[:,1]
table = pd.DataFrame({"P(Y=1|X)":PY1X,"P(Y=0|X)":P1_Y0})
print(table[:2])

#Evolution of Classification Algo
from sklearn.metrics import accuracy_score,jaccard_similarity_score
jss=jaccard_similarity_score(Ya,Yp)
acc=accuracy_score(Ya,Yp)
print("jss is {} and acc is {}".format(jss,acc))

from sklearn.metrics import confusion_matrix,classification_report
#Confusion matrix
my_cm = confusion_matrix(Ya,Yp)
print(my_cm)
table1 = pd.DataFrame({"Ya":Ya,"Yp":Yp})
CONMAT = pd.crosstab(table1.Ya,table1.Yp,rownames=["Yactual"],colnames=["Predicted"],margins=True)
sb.heatmap(CONMAT,annot=True)
plt.show()
c_class =float( my_cm[0,0]+my_cm[1,1])
print(c_class)
ts=float(my_cm.sum())
print(ts)
cx=float(c_class/ts)
print(cx)
plt.matshow(my_cm)
plt.title("this is confusion matrix")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()

myclassreport = classification_report(Ya,Yp)
print(myclassreport)

from sklearn.metrics import roc_curve
Ypprr=Ypprob[:,1]
fpr,tpr,th=roc_curve(Ya,Ypprr)
print(tpr)
plt.plot(fpr,tpr,label="ROC Curve")
plt.show()


