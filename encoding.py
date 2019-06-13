import pandas as pd 
mydata = pd.read_csv("Car_sales.csv")
#print(mydata.isnull().sum())

import numpy as np
from sklearn.preprocessing import LabelEncoder
#print(np.unique(mydata["Manufacturer"]))
qm=np.unique(mydata.iloc[:,0])
car=mydata.iloc[:,0]
print(car.value_counts())

import seaborn as sb 
from matplotlib import pyplot as plt 
sb.countplot("Manufacturer",data=mydata)
plt.show()
le=LabelEncoder()
mydata.iloc[:,0]=le.fit_transform(mydata.iloc[:,0])
print(mydata)
