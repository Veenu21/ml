#Slicing the string
x='Codevita'
y1=""
y=[]
i=0
t=3
for i in range(3):
    Lx=x[:-1]
    print("Lx"+ Lx)
    Rx=x[-1]
    print("Rx"+ Rx)
    x1=Rx+Lx
    print("x1"+x1)
    y.append(Rx)
y1.join(y)    
def check(x,y):
  print(y1)
  if(x.find(y1) == -1):
    print("YES")
  else:
    print("NO")
check(x,y)    