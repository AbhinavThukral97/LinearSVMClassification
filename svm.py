"""
Title: Detection of Forest Cover Type using Linear SVM
Author: Abhinav Thukral
Description: 
Built Linear SVM from scratch in Python 3.5 for multi-class classification of forest cover types.
Implementation of Linear SVM (Support Vector Machine Classification), Gradient Descent, one v/s all classification, feature scaling and cross validations from scratch.
"""

import pandas as pd
from random import seed
from random import randrange
from math import exp
from math import log
from math import floor

#Function to split data into train and test
def cross_val_split(data_X,data_Y,test_size,seed_val):
	data_x = data_X.tolist()
	data_y = data_Y.tolist()
	seed(seed_val)
	train_size = floor((1 - test_size)*len(data_x))
	train_x = []
	train_y = []
	while(len(train_x)<train_size):
		index = randrange(len(data_x))
		train_x.append(data_x.pop(index))
		train_y.append(data_y.pop(index))
	return train_x,train_y,data_x,data_y

#Function to return columnwise max-min statistics for scaling
def statistics(x):
	cols = list(zip(*x))
	stats = []
	for e in cols:
		stats.append([min(e),max(e)])
	return stats

#Function to scale the features
def scale(x, stat):
	for row in x:
		for i in range(len(row)):
			row[i] = (row[i] - stat[i][0])/(stat[i][1] - stat[i][0])

#Function to convert different classes into different columns to implement one v/s all
def one_vs_all_cols(s):
	m = list(set(s))
	m.sort()
	for i in range(len(s)):
		new = [0]*len(m)
		new[m.index(s[i])] = 1
		s[i] = new
	return m

#Function to compute Theta transpose x Feature Vector
def ThetaTX(Q,X):
	det = 0.0
	for i in range(len(Q)):
		det += X[i]*Q[i]
	return det

#Function to compute cost for negative class (classs = 0)
def LinearSVM_cost0(z):
	if(z < -1): #Ensuring margin
		return 0
	return z + 1

#Function to compute cost for positive class (classs = 1)
def LinearSVM_cost1(z):
	if(z > 1): #Ensuring margin
		return 0
	return -z + 1

#function to calculate sigmoid
def sigmoid(z):
	return 1.0/(1.0 + exp(-z))

#Function to calculate SVM cost
def cost(theta,c,x,y):
	cost = 0.0
	for i in range(len(x)):
		z = ThetaTX(theta[c], x[i])
		cost += y[i]*LinearSVM_cost1(z) + (1 - y[i])*LinearSVM_cost0(z)
		#cost += -1*(y[i]*log(sigmoid(z)) + (1 - y[i])*log(1 - sigmoid(z)))
	return cost

#Function to perform Gradient Descent on the weights/parameters
def gradDescent(theta,c,x,y,learning_rate):
	oldTheta = theta[c]
	for Q in range(len(theta[c])):
		derivative_sum = 0 
		for i in range(len(x)):
			derivative_sum += (sigmoid(ThetaTX(oldTheta,x[i])) - y[i])*x[i][Q]
		theta[c][Q] -= learning_rate*derivative_sum

#Function to return predictions using trained weights
def predict(data,theta):
	predictions = []
	count = 1
	for row in data:
		hypothesis = []
		multiclass_ans = [0]*len(theta)
		for c in range(len(theta)):
			z = ThetaTX(row,theta[c])
			hypothesis.append(sigmoid(z))
		index = hypothesis.index(max(hypothesis))
		multiclass_ans[index] = 1
		predictions.append(multiclass_ans)
		count+=1
	return predictions

#Function to return accuracy
def accuracy(predicted, actual):
	n = len(predicted)
	correct = 0
	for i in range(n):
		if(predicted[i]==actual[i]):
			correct+=1
	return correct/n

#Function to perform cross validation
def cross_validation(x,y,test_data_size,validations,learning_rate,epoch):
	print("No. of validation checks to be performed: ",validations)
	print("No. of Iterations per validation: ",epoch)
	accuracies = []
	for valid in range(validations):
		print("\nRunning Validation",valid+1)
		x_train, y_train, x_test, y_test = cross_val_split(x,y,test_data_size,valid+1)
		#Converting y_train to classwise columns with 0/1 values
		classes = []
		for i in range(len(label_map)):
			classes.append([row[i] for row in y_train])
		#Initialising Theta (Weights)
		theta = [[0]*len(x_train[0]) for _ in range(len(classes))]
		#Training the model
		for i in range(epoch):
			for class_type in range(len(classes)):
				gradDescent(theta,class_type,x_train,classes[class_type],learning_rate)
			if(i%(epoch/10)==0):
				print("Processed", i*100/epoch,"%")
		print("Completed")
		#Predicting using test data
		y_pred = predict(x_test,theta)
		#Calculating accuracy
		accuracies.append(accuracy(y_pred,y_test))
		print("Validation",valid+1,"accuracy score: ",accuracies[valid])
	return sum(accuracies)/len(accuracies)

#Dataset url to be imported
print("Running Forest Cover Detection using Linear SVM\n")
url = "dataset.csv"
dataset = pd.read_csv(url)
data = dataset.values
#Assigning x and y - features and classes
x = data[:,:26]
y = data[:,27]
#Feature Scaling by using column wise max, min stats
stats = statistics(x)
scale(x,stats)
#Converting different labels to columns
#label_map can be used later to retrieve the predicted class label in the original form (string format)
label_map = one_vs_all_cols(y)
#Splitting dataset into training and testing data
test_data_size = 0.2
learning_rate = 0.01
epoch = 500
validations = 5
final_score = cross_validation(x,y,test_data_size,validations,learning_rate,epoch)
#Printing Final Stats
print("\nReport")
print("Model used: ","Linear SVM using Gradient Descent")
print("Learning rate: ", learning_rate)
print("No. of iterations: ",epoch)
print("No. of features: ", len(x[0]))
print("Training data size: ", floor(len(x)*(1 - test_data_size)))
print("Test data size: ", len(x) - floor(len(x)*(1 - test_data_size)))
print("No. of validation tests performed: ", validations)
print("Accuracy: ",final_score*100,"%")