# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Load the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary. 6.Define a function to predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Placement_Data.csv')
dataset

dataset = dataset.drop('sl_no',axis=1)
dataset = dataset.drop('salary',axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y

theta = np.random.randn(X.shape[1])
y =Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h= sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta

theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations = 1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred

y_pred = predict(theta,X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(y_pred)

print(Y)

xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```
## Output:

# Read the file and Display
<img width="1427" height="524" alt="Screenshot 2025-09-27 105650" src="https://github.com/user-attachments/assets/1606d432-7af3-44ec-b6b9-701d6f4031d1" />

## Categarizing the Columns
<img width="378" height="596" alt="Screenshot 2025-09-27 105659" src="https://github.com/user-attachments/assets/59b4be42-6958-4628-986b-9c86fceeb76d" />

## Labelling columns and displaying dataset
<img width="1162" height="532" alt="Screenshot 2025-09-27 105707" src="https://github.com/user-attachments/assets/44bcb032-03e2-4762-b6e8-542a4cdd01e4" />

## Display dependent variable
<img width="821" height="234" alt="Screenshot 2025-09-27 105715" src="https://github.com/user-attachments/assets/6cf1e5e4-54b1-494b-9efd-5e27096309f8" />

## Accuracy
<img width="879" height="313" alt="Screenshot 2025-09-27 105725" src="https://github.com/user-attachments/assets/557805d2-c658-4d56-8a32-8311326a5cdc" />

## Printing Y
<img width="393" height="67" alt="Screenshot 2025-09-27 105731" src="https://github.com/user-attachments/assets/9920243b-7ca7-49e3-8692-91dde8212d27" />



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

