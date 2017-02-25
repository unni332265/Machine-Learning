#!/usr/bin/python
import random
import math
import matplotlib.pyplot as plt	
import numpy as np
from numpy.linalg import inv

f = open("Winequality_dataset.csv", "r")

data =[]

train_data = []
test_data = []

for i in range(6498):
	data.append(f.readline().strip().split(','))

data.pop(0)

for datum in data[0:4500]:
	train_data.append(datum)

for datum in data[4501:6497]:
	test_data.append(datum)

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for datum in train_data:
	x_train.append(datum[:12])
	y_train.append(datum[12:13])

x=np.matrix(x_train).astype("float")
y=np.matrix(y_train).astype("float")
x1=np.matrix(x_train).astype("float")
y1=np.matrix(y_train).astype("float")
y_t=np.transpose(y)

for datum in test_data:
	x_test.append(datum[:12])
	y_test.append(datum[12:13])

for i in range(12):
	w.append(random.random())
n_i=50
eta=0.01
x_t=[]
w_transpose=np.transpose(w)

for n_m in range(n_i):
	gr=[]
	s=[]
	for i in range(12):
		s.append(0)
		gr.append(0)
	for i in range(4500):
		x_t=np.transpose(x[i])
		h=y[i]*x[i]
		a=(-1)*w_transpose*x_t
		s=x[i]/(1+math.exp(a))
		gr=h-s
		for i in range(12):
			w[i]=w[i]+eta*gr

err_out=[]
error_out=[]
inp=[500,1500,2500,3500,4500]	
for j in inp: 			
	e_out=[]
	for i in range(len(test_data)):
		x_t=np.transpose(x1[i])
		a=-y1[i]*w_transpose*x_t
		e_out.append(math.log(1+math.exp(a)))
	error_out.append(sum(e_out)/len(test_data))
err_out=list(np.array(error_out).flat)	

err_in=[]
error_in=[]
for inp1 in inp:
	e_in=[]
	for i in range(inp1):
		x_t=np.transpose(x[i])
		a=-y[i]*w_transpose*x_t
		e_in.append(math.log(1+math.exp(a)))
	error_in.append(sum(e_in)/inp1)
err_in=list(np.array(error_in).flat)

plt.plot(inp,err_out)
plt.plot(inp,err_in)
plt.ylabel('Error')
plt.xlabel('Number of Inputs')
plt.title("Gradient Descent")
plt.show()