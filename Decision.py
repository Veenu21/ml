import pandas as pd 

from matplotlib import pyplot as plt 
import seaborn as sb 

mydata=pd.read_csv("drug_dt.csv")
print(mydata.shape)
print(mydata.dtypes)
print(mydata.head(1))
#sb.countplot(x='Drug',hue='Sex',data=mydata)
#plt.show()
print(mydata.Drug.value_counts())
X=mydata.iloc[:,0:5]
Y=mydata.iloc[:,5]
print(X.head)
X.Sex[x.Sex=='M']=0
X.Sex[x.Sex=='F']=1
X.BP[x.BP=='LOW']=0
X.BP[x.BP=='HIGH']=2
X.BP[x.BP=='NORMAL']=1
X.Cholesterol[x.Cholesterol=='HIGH']=2
X.Cholesterol[x.Cholesterol=='HIGH']=1
X.Cholesterol[x.Cholesterol=='HIGH']=0
#import pydotplus
from sklearn import trees