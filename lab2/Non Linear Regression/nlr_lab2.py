#!/usr/bin/python
import random
import matplotlib.pyplot as plt	
import numpy as np
from numpy.linalg import inv
import itertools

f = open("winequality-white.csv", "r")
data =[]

train_data = []
test_data = []

for i in range(4899):
	data.append(f.readline().strip().split(';'))
data.pop(0)
random.Random(1234).shuffle(data)
float_data=[]

for datum in data:
	float_data.append(np.array(datum).astype(np.float))

data1=[]
nl_data=[]

m=0
for i in range(4898):
	new=[]
	for j in range(11):
		for k in range(j,11):
			new.append(float(data[i][j])*float(data[i][k]))
			m=m+1;
	data1.append(new)

data2=[]
for i in range(4898):
	new=[]
	for j in range(1):
		new.append(1)
	data2.append(new)

nl_data=np.hstack((data1,data2,data))
inp=int(raw_input('Enter the input:'))
n=(inp*4898)/100
for datum in nl_data[0:n]:
	train_data.append(datum)

for datum in nl_data[n:4898]:
	test_data.append(datum)

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for datum in train_data:
	x_train.append(datum[:78])
	y_train.append(datum[78:79])


z=np.matrix(x_train).astype("float")
x_transpose = np.transpose(z)


for datum in test_data:
	x_test.append(datum[:78])
	y_test.append(datum[78:79])

xplus=inv(x_transpose*z)*x_transpose
y_train = np.matrix(y_train).astype("float")
w=xplus*y_train
error=[]
w_transpose=np.transpose(w)
m=np.matrix(x_test).astype("float")
y_test = np.matrix(y_test).astype("float")
k=m*w-y_test
error=np.square(k)
error_sum=sum(error)
print error_sum/n
inp=[30,40,50,60,70]
final_error=[1.30228358,0.82804418,0.58269823,0.58269823,0.23140514]
plt.plot(inp,final_error)
plt.xlabel('Number of inputs')
plt.ylabel('Normalised Error')
plt.title('Analysis of Non-Linear Regression model')
plt.show()