my_a = {"Name" : ["x","y","z"], "Age": [11,12,13]}
print(my_a)
print(my_a.keys())
print(my_a.values())
import pandas as pd 
a = ['a','b','c']
a_s = pd.Series(a,index=["one","two","three"])
print(a_s["one"])
a_di = pd.DataFrame(my_a)
print(a_di)
print(a_di.ndim)
print(a_di.shape)