import pandas as pd 
import numpy as np 
X_feature = np.array([[1],[2],[3],[4],[5]])
Y_feature = np.array([[1],[3],[2],[3],[5]])

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
teacher = LinearRegression()
learner = teacher.fit(X_feature,Y_feature)
print(teacher)
Yp=learner.predict(X_feature)
c=learner.intercept_
m=learner.coef_
print("c is{} m is {} and Yp is {}".format(c,m,Yp))
xlist = list(X_feature)
ylist = list(Y_feature)
yplist = list(Yp)

mytable =pd.DataFrame({"Input": xlist,"Output": ylist , "Yp" : yplist})
print(mytable)

plt.plot(X_feature,Y_feature,'r*-')
plt.xlabel("input")
plt.ylabel("Ya")
plt.plot(X_feature,Yp,'g*-')
plt.xlabel("input")
plt.ylabel("Ya")
plt.legend(["Ya","Yp"])
plt.show()

from sklearn.metrics import mean_squared_error
Error = mean_squared_error(Yp,Y_feature)
E=np.sqrt(Error)
print(np.sqrt(Error))

Y = m*X_feature+c+E
print(Y)

plt.plot(X_feature,Y_feature,'g*')
plt.plot(X_feature,Y,'r*-')
plt.show()