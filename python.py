import numpy as np
#One D Array
a=[11,1,2,33,44,66,77,10,10]
my_id=np.array(a)
print(type(my_id))
print(my_id.ndim)
print(my_id.shape)
print(my_id.dtype)
print(my_id.itemsize)
#Two D Array
a_2d=np.array([[11,12,13],[10,9,8],[0,19,20]],dtype=float)
print(a_2d)
print(a_2d.shape)
print(len(a_2d))
print(a_2d.size)
print(a_2d[0:2,1:3])
print(a_2d[0:3:2,0:3:2])
print(a_2d[::-1,:])