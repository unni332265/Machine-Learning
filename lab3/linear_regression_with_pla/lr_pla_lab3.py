#!/usr/bin/python
import random
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

f = open("Winequality_dataset.csv", "r")
data =[]

train_data = []
test_data = []

for i in range(6498):
	data.append(f.readline().strip().split(','))

#for datum in data: print datum
data.pop(0)
#random.Random(1234).shuffle(data)
for datum in data[0:4500]:
	train_data.append(datum)

for datum in data[4501:len(data)]:
	test_data.append(datum)



x_train=[]
y_train=[]
x_test=[]
y_test=[]

for datum in train_data:
	x_train.append(datum[:12])
	y_train.append(datum[12:13])


z=np.matrix(x_train).astype("float")
x_transpose = np.transpose(z)


for datum in test_data:
	x_test.append(datum[:12])
	y_test.append(datum[12:13])


xplus=inv(x_transpose*z)*x_transpose
y_train = np.matrix(y_train).astype("float")
weight=xplus*y_train

for i in range(len(y_train)):
	if y_train[i][0]==0:
		y_train[i][0]=-1

for i in range(len(y_test)):
	if y_test[i][0]==0:
		y_test[i][0]=-1

for i in weight:
	print i

n_i=100

inp=int(raw_input())

sum_e_test=[]
sum_e_train_test=[]
count_mp_train=[]
count_mp_test=[]
count_mp_train_test=[]
for num in range(n_i):
	prediction=[]
	count2=0
	count1=0
	count3=0
	for i in range(len(x_train)):
		activate=0.0
		for m in range(12):
			activate+=weight[m]*float(x_train[i][m])
		if activate>=0.0:
			y=1
		else:
			y=-1
		prediction.append(y)
	
	
	for i in range(len(x_train)):
		
		if float(prediction[i])!=float(y_train[i][0]):
			count1=count1+1;
			for j in range(12):
				weight[j]=weight[j]+float(y_train[i][0])*float(x_train[i][j])
	count_mp_train.append(count1)


	prediction_test=[]
	sum_error=0.0
	for i in range(len(x_test)):
		activate=0.0
		for m in range(12):
			activate+=weight[m]*float(x_test[i][m])
		if activate>=0.0:
			y=1
		else:
			y=-1
		prediction_test.append(y)

	for i in range(len(x_test)):
		k=float(y_test[i][0])-prediction_test[i]
		
		sum_error+=abs(k)
		if float(prediction_test[i])!=float(y_test[i][0]):
			count2=count2+1;
			
	count_mp_test.append(count2)
	prediction_train_test=[]
	sum_error=0.0
	for i in range(inp):
		activate=0.0
		for m in range(12):
			activate+=weight[m]*float(x_train[i][m])
		if activate>=0.0:
			y=1
		else:
			y=-1
		prediction_train_test.append(y)

	for i in range(inp):
		sum_error+=abs(k)
		if float(prediction_train_test[i])!=float(y_train[i][0]):
			count3=count3+1;
			
	count_mp_train_test.append(count3)

print count_mp_test
print count_mp_train_test
t=np.arange(0, n_i, 1)
plt.plot(t, count_mp_test)
plt.plot(t, count_mp_train_test,'r')
plt.ylabel('Error')
plt.xlabel('Number of iterations')
plt.title("Linear Regression with Peceptron Learning Algorithm")
plt.show()