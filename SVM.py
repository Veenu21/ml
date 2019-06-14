from sklearn import datasets
import numpy as np 
mydata=datasets.load_digits()
print(type(mydata))
print(mydata.keys())
print(mydata.data.shape)
print(np.unique(mydata.target))
print(mydata.target_names)
#print(mydata.DESCR)
#X1=mydata.data[0]
#print(X1.shape)
#print(X1)
#im=mydata.images[1]
#print(im.shape)
#im[0,0]

from matplotlib import pyplot as plt 
#plt.matshow(im)
#plt.show()
X_input = mydata.data 
Y_target = mydata.target
print("Shape of input {} and shape of target {} ".format(X_input.shape,Y_target.shape))

#split 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_input,Y_target,test_size=.30,random_state=101)
print("Shape of X_train {}  and shape of X_test {}".format(X_train.shape,X_test.shape))
print("Shape of Y_train {}  and shape of Y_test {}".format(Y_train.shape,Y_test.shape))

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score,jaccard_similarity_score
trainer = SVC(gamma=.001)
learner = trainer.fit(X_input,Y_target)
#predict
Ya=Y_test
Yp=learner.predict(X_test)
acc = accuracy_score(Ya,Yp)
jss = jaccard_similarity_score(Ya,Yp)
con = confusion_matrix(Ya,Yp)
cr = classification_report(Ya,Yp)
print("acc is {} and jss is {}".format(acc,jss))
print(con)

from matplotlib import pyplot as plt 
from scipy import misc
mydigit = misc.imread(r"8.jpg")
print(mydigit.shape)
#change digit image into 8*8
mydigit_88  = misc.imresize(mydigit,(8,8))
print("Shape of mydidit {}".format(mydigit_88.shape))
print("Datatype of mydigit_88 {}".format(mydigit_88.dtype))
#Data type conversion of image
mydigit_float = mydigit_88.astype(X_input.dtype)
print("Datatype of mydigit {}".format(mydigit_float.dtype))

#Scaling...down require data of float type and returns data of int type...so we have to convert its dtype again
mydigit_scale=misc.bytescale(mydigit_float,high=16,low=0)
mydigit_newdtype=mydigit_scale.astype(X_input.dtype)
print("Dtype of mydigit new {}".format(mydigit_newdtype.dtype))

##Input data is 2D and our image is 3D so we need to convert
X_testimage = []
for eachrow in mydigit_newdtype:
    for eachpixel in eachrow:
        X_testimage.append(sum(eachpixel)/3.0)
print(len(X_testimage))        
image_array = np.array(X_testimage)
print(learner.predict([image_array]))