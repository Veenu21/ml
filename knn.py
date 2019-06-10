import numpy as np
#input.... feature
x_feature = np.array([[167,51],[182,62],[176,69],[173,64],[172,65],[174,56],[161,58],[173,57],[170,55]])
print(x_feature)
X= x_feature
print(x_feature.shape)
Y_target = np.array(["uw","NO","NO","NO","NO","uw","NO","NO","NO"])
print(Y_target.shape)
from sklearn.neighbors import KNeighborsClassifier
trainer = KNeighborsClassifier(n_neighbors=5)
learner = trainer.fit(x_feature , Y_target)
print(learner.predict([[170,57]]))
from sklearn.metrics import euclidean_distances
#Y = np.array([[170,57]])
#dis = euclidean_distances(X,Y)
#print(dis)