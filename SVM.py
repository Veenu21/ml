from sklearn import datasets
import numpy as np 
mydata=datasets.load_digits()
print(type(mydata))
print(mydata.keys())
print(mydata.data.shape)
print(np.unique(mydata.target))
print(mydata.target_names)
#print(mydata.DESCR)
X1=mydata.data[0]
print(X1.shape)
print(X1)
im=mydata.images[1]
print(im.shape)
im[0,0]

from matplotlib import pyplot as plt 
plt.matshow(im)
plt.show()



