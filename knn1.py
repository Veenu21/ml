import numpy as np 
#input feature....
X_feature = np.array([[1, 9],[2,8],[3,7],[2,8],[6,4],[9,1],[8,2],[9,1],[1,1],[5,5]])
print(X_feature)
print(X_feature.shape)
# target
Y_target = np.array(["Sour","Sour","Sour","Sour","Sweet","Sweet","Sweet","Sweet","None","Sour"])
print(Y_target)
print(Y_target.shape)
from sklearn.neighbors import KNeighborsClassifier
trainer = KNeighborsClassifier(n_neighbors=1)
learner = trainer.fit(X_feature , Y_target)
print(learner.predict([[1,1]]))

