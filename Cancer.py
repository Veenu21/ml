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

#Import StandarScaler from sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

#Import seaborn as sb
import seaborn as sb

#Collect data
mydata = pd.read_excel("breast-cancer-wisconsin.xlsx",names = ["Sample_code_number","Clump_Thickness","Uniformity_of_Cell_Size","Uniformity_of_Cell_Shape","Marginal_Adhesion","Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses","Class"],na_values=["?"])
print(mydata)
names = ["Sample_code_number","Clump_Thickness","Uniformity_of_Cell_Size","Uniformity_of_Cell_Shape","Marginal_Adhesion","Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses","Class"]
print(mydata.isnull().sum())
#print(mydata.describe().T)
#print(mydata.corr())
#mydata.hist(bins=50)
#plt.show()
mydata.fillna(mydata["Bare_Nuclei"].mean(),inplace=True)
X_input= mydata[["Sample_code_number","Clump_Thickness","Uniformity_of_Cell_Size" ,"Uniformity_of_Cell_Shape","Marginal_Adhesion","Single_Epithelial_Cell_Size","Bare_Nuclei","Bland_Chromatin","Normal_Nucleoli","Mitoses"]]
print(X_input.head())
Y_output= mydata["Class"]
print("type of X_data is{} and type of Y_data is {} ".format(type(X_input),type(Y_output)))

#Split the data in train and test
X_train,X_test,Y_train,Y_test = train_test_split(X_input,Y_output,test_size=.30,random_state=101)
print("Sample of X_train {} and X_test {}".format(len(X_train),len(X_test)))
print("Sample of Y_train {} and Y_test {}".format(len(Y_train),len(Y_test)))

#Preprocessing
#Down feature scaling(preprocessing)
myscaler=StandardScaler()
myscaler.fit(X_train)
print(type(X_train))
X_train=myscaler.transform(X_train)
print(type(X_train))
X_test=myscaler.transform(X_test)
#XA[:]
XA_mean=X_train.mean()
XA_std=X_train.std()
print("mean of XA {} and std of XA {}".format(abs(round(XA_mean)),XA_std))

#Evolution of Classification Algo

#Total no. of class
print(np.unique(Y_output))

#Model
from sklearn.neural_network import MLPClassifier
Trainer = MLPClassifier(hidden_layer_sizes=(11,11,11),activation='relu',max_iter=200)
#print(Trainer)
learner =Trainer.fit(X_train,Y_train)
#Testing
Yp=learner.predict(X_test)
YA=Y_test
Yplist=list(Yp)
Yalist=list(YA)
table=pd.DataFrame({"Ya":Yalist,"Yp":Yplist})
print(table[:1])


#Calculation of accuracy score ,jaccardsimilarity,confusion matrix
from sklearn.metrics import accuracy_score,jaccard_similarity_score
jss=jaccard_similarity_score(Yp,YA)
acc=accuracy_score(Yp,YA)
print("jss is {} and acc is {}".format(jss,acc))


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(YA,Yp)
print(cm)
report=classification_report(YA,Yp)
print(report)
print(learner.intercepts_)
lb=learner.intercepts_[0]
print(lb)
lw=learner.coefs_[0]
print(lw)
table1 = pd.DataFrame({"Ya":YA,"Yp":Yp})
CONMAT = pd.crosstab(table1.Ya,table1.Yp,rownames=["Yactual"],colnames=["Predicted"],margins=True)
sb.heatmap(CONMAT,annot=True)
plt.show()
c_class =float( cm[0,0]+cm[1,1])
print(c_class)
ts=float(cm.sum())
print(ts)
cx=float(c_class/ts)
print(cx)
plt.matshow(cm)
plt.title("this is confusion matrix")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()


