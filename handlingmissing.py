import pandas as pd 
mydata=pd.read_csv("Titanic_Data.csv")
#print(mydata)
#print(mydata.isnull().sum())
X=mydata
xm=mydata.dropna(thresh=2)
print("Shape of X {} and shape of ym {}".format(X.shape,xm.shape))
#xm=mydata.dropna(inplace=true)
mydata.fillna(mydata["Parch"].mean(),inplace=True)
print(mydata.isnull().sum())
#groupby
print(mydata.keys())
print(mydata.groupby(["Survived","Sex"]).mean())

