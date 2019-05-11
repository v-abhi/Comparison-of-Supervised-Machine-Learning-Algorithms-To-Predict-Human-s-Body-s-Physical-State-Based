#!/usr/bin/env python

# for file handling
import os

# importing diffrent models
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# for splitting data into trained and tested set
from sklearn.model_selection import train_test_split

# for saving generated model
from sklearn.externals import joblib

# for reading csv file
import csv

# for finding accuracy of model
from sklearn.metrics import accuracy_score


## for creating confusion matrix
from sklearn import metrics


# for detailing about models
from sklearn.metrics import classification_report,f1_score


# for graphing
import matplotlib.pyplot as plt
    
    
import pandas as pd

import numpy as np

    
    



from sklearn import datasets
import numpy
## importing data
iris = datasets.load_iris()




# importing data
def import_data(file):
    file_dir = os.getcwd()
    file_path= os.path.join(file_dir,'data/'+file)            ### Make sure the dataset is in the correct folder
    train_data = csv.reader(file_path)
    #train_data = pd.read_csv(file_path, sep=',',header=None)
    return train_data
    
    
    
    
# models





acc = np.zeros((2, 9))





    
    
def Logistic_Regression(X_train,y_train,X_test,y_test,k):
    clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    joblib.dump(clf, 'log_reg.pkl')
    print("Logistic_Regression accuracy",end='  ')
    print(accuracy)
    print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
    print("\nconfusion matrix\n")
    print(cnf_matrix)
    plt.imshow(cnf_matrix)
    plt.colorbar()
    plt.title("Confusion Matrix of Logistic Regression\nAccuracy = "+str(accuracy))
    plt.xlabel('predictive value')
    plt.ylabel('actual value')
    for i in range(len(cnf_matrix[0])):
        for j in range(len(cnf_matrix[0])):
            plt.text(j,i,str(cnf_matrix[i][j]))
    plt.savefig('Logistic_Regression.jpg')
    #plt.show()
    print("\n")
    print("\n")
    print(j)
    acc[0][k-1] = accuracy

    
    
    
    
    
def Naive_Bayes(X_train,y_train,X_test,y_test,k):
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    joblib.dump(clf, 'Gaussian.pkl')
    print("Naive_Bayes accuracy",end='  ')
    print(accuracy)
    print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
    print("\nconfusion matrix\n")
    print(cnf_matrix)
    plt.imshow(cnf_matrix)
    plt.colorbar()
    plt.title("Confusion Matrix of Naive Bayes\nAccuracy = "+str(accuracy))
    plt.xlabel('predictive value')
    plt.ylabel('actual value')
    for i in range(len(cnf_matrix[0])):
        for j in range(len(cnf_matrix[0])):
            plt.text(j,i,str(cnf_matrix[i][j]))
    plt.savefig('Naive_Bayes.jpg')
    #plt.show()
    print("\n")
    print("\n")
    acc[1][k-1] = accuracy






# Importing data
#data = import_data('g_s.csv') 


import csv

data = []
with open("g_s.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
    for row in reader: # each row is a list
        data.append(row)

data = np.asarray(data)








# splitting data_set into two parts


for i in range(9):
        

        train_data, test_data = train_test_split(data, test_size=(0.1*(i+1)), random_state=42)



        # assigning features space and lables space
        X_train = train_data[:,1:4]
        y_train = train_data[:,4]

        X_test = test_data[:,1:4]
        y_test = test_data[:,4]








        
        NeuralNetworkTrain(X_train,y_train,X_test,y_test,i+1)
        Logistic_Regression(X_train,y_train,X_test,y_test,i+1)
        Naive_Bayes(X_train,y_train,X_test,y_test,i+1)
        Stochastic_Gradient_Descent(X_train,y_train,X_test,y_test,i+1)
        K_Nearest_Neighbours(X_train,y_train,X_test,y_test,i+1)
        Decision_Tree(X_train,y_train,X_test,y_test,i+1)
        Random_Forest(X_train,y_train,X_test,y_test,i+1)
        Support_Vector_Machine(X_train,y_train,X_test,y_test,i+1)


print(acc)










