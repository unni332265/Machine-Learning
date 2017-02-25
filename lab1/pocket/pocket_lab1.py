#!/usr/bin/python
#implementation of the pocket algorithm
import random
import matplotlib.pyplot as plt	
import numpy as np

f = open("iris_data_binary.txt", "r")
data =[]

for i in range(150):
	data.append(f.readline().strip().split('\t'))
random.Random(1234).shuffle(data)
train_data = []
test_data = []
inp=int(raw_input('Enter the input:'))
n=(inp*150)/100

for datum in data[0:n]:
	train_data.append(datum)

for datum in data[n:150]:
	test_data.append(datum)
weight = []

for i in range(4):
	weight.append(random.random())

print weight
no_iterations=100
sum_e_test=[]
sum_e_train_test=[]
count_mp_train=[]
count_mp_test=[]
count_mp_train_test=[]
min_e_test=[]
min_e_train_test=[]

for num in range(no_iterations):
	prediction=[]
	count2=0
	count1=0
	count3=0
	#predicting the output for the train data
	for i in range(len(train_data)):
		activate=0.0
		for m in range(4):
			activate+=weight[m]*float(train_data[i][m])
		if activate>=0.0:
			y=1
		else:
			y=-1
		prediction.append(y)
	#checking whether the prediction is right
	for i in range(len(train_data)):
		if float(prediction[i])!=float(train_data[i][4]):
			#if wrong, increment the count and update the weights
			count1=count1+1;
			#updating the weights if the prediction is wrong
			for j in range(4):
				weight[j]=weight[j]+float(train_data[i][4])*float(train_data[i][j])
	count_mp_train.append(count1)
	#Using the weights,predicting the error in the test data
	prediction_test=[]
	sum_error=0.0
	#predicting the output for the test data
	for i in range(len(test_data)):
		activate=0.0
		for m in range(4):
			activate+=weight[m]*float(test_data[i][m])
		if activate>=0.0:
			y=1
		else:
			y=-1
		prediction_test.append(y)
	for i in range(len(test_data)):
		k=float(test_data[i][4])-prediction_test[i]
		sum_error+=abs(k)
		#checking whether the prediction was right
		#if wrong,increment the count
		if float(prediction_test[i])!=float(test_data[i][4]):
			count2=count2+1;
	count_mp_test.append(count2)
	sum_e_test.append(sum_error/len(test_data))
	#Using the weights,predicting the error in the train data
	prediction_train_test=[]
	sum_error=0.0
	#predicting the output for the train data
	for i in range(len(train_data)):
		activate=0.0
		for m in range(4):
			activate+=weight[m]*float(train_data[i][m])
		if activate>=0.0:
			y=1
		else:
			y=-1
		prediction_train_test.append(y)

	for i in range(len(train_data)):
		k=float(train_data[i][4])-prediction_train_test[i]
		sum_error+=abs(k)
		#checking whether the prediction was right
		#if wrong,increment the count
		if float(prediction_train_test[i])!=float(train_data[i][4]):
			count3=count3+1;
			
	count_mp_train_test.append(count3)
	sum_e_train_test.append(sum_error/len(train_data))

#finding out the minimum error in Eout
min_error=sum_e_test[0]
for num in range(no_iterations):
	if sum_e_test[num]<min_error:
		min_error=sum_e_test[num]
		min_e_test.append(min_error)
	else:
		min_e_test.append(min_error)

#finding out the minimum error in Ein 
min_error=sum_e_train_test[0]
for num in range(no_iterations):
	if sum_e_train_test[num]<min_error:
		min_error=sum_e_train_test[num]
		min_e_train_test.append(min_error)
	else:
		min_e_train_test.append(min_error)

print count_mp_train
print count_mp_test
print count_mp_train_test
print min_e_test
print min_e_train_test

t = np.arange(0, no_iterations, 1)
plt.plot(t, min_e_test)
plt.plot(t, min_e_train_test,'r')
plt.ylabel('Error')
plt.xlabel('Number of iterations')
plt.title("Pocket Algorithm 70/30")
plt.show()