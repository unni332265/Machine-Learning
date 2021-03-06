#!/usr/bin/python
#implemetation of linear regression on the given dataset
import random
import matplotlib.pyplot as plt	
import numpy as np
from numpy.linalg import inv

f = open("winequality-white.csv", "r")
data =[]

train_data = []
test_data = []

for i in range(4899):
	data.append(f.readline().strip().split(';'))

data.pop(0)
random.Random(1234).shuffle(data)
inp=int(raw_input('Enter the input:'))
n=(inp*4898)/100
for datum in data[0:n]:
	train_data.append(datum)
for datum in data[n:4898]:
	test_data.append(datum)

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for datum in train_data:
	x_train.append(datum[:11])
	y_train.append(datum[11:12])

z=np.matrix(x_train).astype("float")
x_transpose = np.transpose(z)

for datum in test_data:
	x_test.append(datum[:11])
	y_test.append(datum[11:12])

#implementing Linear Regression 

xplus=inv(x_transpose*z)*x_transpose
y_train = np.matrix(y_train).astype("float")
w=xplus*y_train
error=[]
w_transpose=np.transpose(w)
m=np.matrix(x_test).astype("float")
y_test =np.matrix(y_test).astype("float")
k=m*w-y_test
error=np.square(k)
error_sum=sum(error)
print error_sum/n
inp=[30,40,50,60,70]
final_error=[1.3213091,0.84423266,0.56810426,0.38306507,0.24643138]
plt.plot(inp,final_error)
plt.xlabel('Number of inputs')
plt.ylabel('Normalised Error')
plt.title('Analysis of Linear Regression model')
plt.show()