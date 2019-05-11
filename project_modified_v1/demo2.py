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
    
    
    
    
#
acc = np.zeros((8, 9))
    
    
# models



def NeuralNetworkTrain(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            for i in 5, 10, 15:                    #### We are taking only one hidden layer, try with different number of layers
                #print("hidden layer: ",i,"\n")
                clf = MLPClassifier(hidden_layer_sizes=(i,i,i),early_stopping=True,learning_rate='adaptive',learning_rate_init=0.003)
                clf.fit(X_train,y_train)

                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                
                
            acc[0][k-1] = accuracy 
    
    if u == 1 and y == 1:
            for i in 5, 10, 15:                    #### We are taking only one hidden layer, try with different number of layers
                #print("hidden layer: ",i,"\n")
                clf = MLPClassifier(hidden_layer_sizes=(i,i,i),early_stopping=True,learning_rate='adaptive',learning_rate_init=0.003)
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)      
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            joblib.dump(clf, 'n_n.pkl')    
            print("Neural Network accuracy",end='  ')
            print(accuracy)
            print("Fitting of test data for size ",i," : \n",classification_report(y_test,y_pred))
            print("\nconfusion matrix\n")
            print(cnf_matrix)
            plt.imshow(cnf_matrix)
            plt.colorbar()
            plt.title("Confusion Matrix of Neural Network\nAccuracy = "+str(accuracy))
            plt.xlabel('predictive value')
            plt.ylabel('actual value')
            for i in range(len(cnf_matrix[0])):
                for j in range(len(cnf_matrix[0])):
                    plt.text(j,i,str(cnf_matrix[i][j]))
            plt.savefig('Neural Network.jpg')
            plt.show()
            print("\n")
            print("\n")

        








    
    
def Logistic_Regression(X_train,y_train,X_test,y_test,k,u):
    
    if u == 0:
            clf = LogisticRegression(solver='lbfgs', multi_class='ovr')
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[1][k-1] = accuracy
   


    if u == 1 and y == 1:
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
            plt.show()
            print("\n")
            print("\n")
    
    
    
    
    
def Naive_Bayes(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            clf = GaussianNB()
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[2][k-1] = accuracy
            
    if u ==1 and y == 1:
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
            plt.show()
            print("\n")
            print("\n")
    
def Stochastic_Gradient_Descent(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100, tol=0.21)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[3][k-1] = accuracy
    
    if u == 1 and y == 1:
    
               
            clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100, tol=0.21)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            joblib.dump(clf, 'sgd.pkl')
            print("Stochastic_Gradient_Descent accuracy",end='  ')
            print(accuracy)
            set(y_test) - set(y_pred)
            print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
            print("\nconfusion matrix\n")
            print(cnf_matrix)
            plt.imshow(cnf_matrix)
            plt.colorbar()
            plt.title("Confusion Matrix of Stochastic Gradient Descent\nAccuracy = "+str(accuracy))
            plt.xlabel('predictive value')
            plt.ylabel('actual value')
            for i in range(len(cnf_matrix[0])):
                for j in range(len(cnf_matrix[0])):
                    plt.text(j,i,str(cnf_matrix[i][j]))
            plt.savefig('Stochastic_Gradient_Descent.jpg')
            plt.show()
            print("\n")
            print("\n")
    
def K_Nearest_Neighbours(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[4][k-1] = accuracy
    
    if u == 1 and y == 1:
            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            joblib.dump(clf, 'knn.pkl')
            print("K_Nearest_Neighbours accuracy",end='  ')
            print(accuracy)
            print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
            print("\nconfusion matrix\n")
            print(cnf_matrix)
            plt.imshow(cnf_matrix)
            plt.colorbar()
            plt.title("Confusion Matrix of K Nearest Neighbours\nAccuracy = "+str(accuracy))
            plt.xlabel('predictive value')
            plt.ylabel('actual value')
            for i in range(len(cnf_matrix[0])):
                for j in range(len(cnf_matrix[0])):
                    plt.text(j,i,str(cnf_matrix[i][j]))
            plt.savefig('K_Nearest_Neighbours.jpg')
            plt.show()
            print("\n")
            print("\n")
    
    
    
def Decision_Tree(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            clf = DecisionTreeClassifier()
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[5][k-1] = accuracy
    
    if u == 1 and y == 1:
            clf = DecisionTreeClassifier()
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            joblib.dump(clf, 'dt.pkl')
            print("Decision_Tree accuracy",end='  ')
            print(accuracy)
            print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
            print("\nconfusion matrix\n")
            print(cnf_matrix)
            plt.imshow(cnf_matrix)
            plt.colorbar()
            plt.title("Confusion Matrix of Decision Tree\nAccuracy = "+str(accuracy))
            plt.xlabel('predictive value')
            plt.ylabel('actual value')
            for i in range(len(cnf_matrix[0])):
                for j in range(len(cnf_matrix[0])):
                    plt.text(j,i,str(cnf_matrix[i][j]))
            plt.savefig('Decision_Tree.jpg')
            plt.show()
            print("\n")
            print("\n")
            
    
    
def Random_Forest(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[6][k-1] = accuracy
    
    if u == 1 and y == 1:
            clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            joblib.dump(clf, 'rf.pkl')
            print("Random_Forest accuracy",end='  ')
            print(accuracy)
            print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
            print("\nconfusion matrix\n")
            print(cnf_matrix)
            plt.imshow(cnf_matrix)
            plt.colorbar()
            plt.title("Confusion Matrix of Random Forest\nAccuracy = "+str(accuracy))
            plt.xlabel('predictive value')
            plt.ylabel('actual value')
            for i in range(len(cnf_matrix[0])):
                for j in range(len(cnf_matrix[0])):
                    plt.text(j,i,str(cnf_matrix[i][j]))
            plt.savefig('Random_Forest.jpg')
            plt.show()
            print("\n")
            print("\n")
    
    
def Support_Vector_Machine(X_train,y_train,X_test,y_test,k,u):
    if u == 0:
            clf = SVC(kernel='linear') # Linear Kernel
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            acc[7][k-1] = accuracy
    
    if u == 1 and y == 1:
            clf = SVC(kernel='linear') # Linear Kernel
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
            joblib.dump(clf, 'svm.pkl')
            print("Support_Vector_Machine accuracy",end='  ')
            print(accuracy)
            print("\nFitting of test data for size : \n\n",classification_report(y_test,y_pred))
            print("\nconfusion matrix\n")
            print(cnf_matrix)
            plt.imshow(cnf_matrix)
            plt.colorbar()
            plt.title("Confusion Matrix of Support Vector Machine\nAccuracy = "+str(accuracy))
            plt.xlabel('predictive value')
            plt.ylabel('actual value')
            for i in range(len(cnf_matrix[0])):
                for j in range(len(cnf_matrix[0])):
                    plt.text(j,i,str(cnf_matrix[i][j]))
            plt.savefig('Support_Vector_Machine.jpg')
            plt.show()
            print("\n")
            print("\n")
    









import csv


for y in range(1,7):


        data = []
        f1 = "d"
        f1 = f1 + str(y)
        f1 = f1 + ".csv"
        with open(f1) as csvfile:
            reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
            for row in reader: # each row is a list
                data.append(row)

        data = np.asarray(data)


        # splitting data_set into two parts


        for i in range(1,10):

                
                u=0
                train_data, test_data = train_test_split(data, test_size=(0.1*(i)), random_state=42)



                # assigning features space and lables space
                X_train = train_data[:,1:4]
                y_train = train_data[:,4]

                X_test = test_data[:,1:4]
                y_test = test_data[:,4]


                NeuralNetworkTrain(X_train,y_train,X_test,y_test,i,u)
                Logistic_Regression(X_train,y_train,X_test,y_test,i,u)
                Naive_Bayes(X_train,y_train,X_test,y_test,i,u)
                Stochastic_Gradient_Descent(X_train,y_train,X_test,y_test,i,u)
                K_Nearest_Neighbours(X_train,y_train,X_test,y_test,i,u)
                Decision_Tree(X_train,y_train,X_test,y_test,i,u)
                Random_Forest(X_train,y_train,X_test,y_test,i,u)
                Support_Vector_Machine(X_train,y_train,X_test,y_test,i,u)


        #print(acc)


        # name of models
        name = ['NeuralNetwork','Logistic_Regression','Naive_Bayes','Stochastic_Gradient','K_Nearest_Neighbours','Decision_Tree','Random_Forest','Support_Vector_Machine']

        # plotting accuracy of different models
        import matplotlib.pyplot as plt
        x = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        plt.xlabel("training_percentage")
        plt.ylabel("accuracy")
        # reading sample size of csv file
        input_file = open(f1,"r+")
        reader_file = csv.reader(input_file)
        value = len(list(reader_file))
        plt.title("sample size = "+str(value))
        for i in range(len(acc)):
            plt.plot(x,acc[i],label = name[i])
        plt.legend()

        filename = "comparasion_chart"
        filename = filename + str(y)
        filename = filename + ".png"

        plt.savefig(filename)
        plt.show()


        print("\n\n")

        ind = np.zeros((1, 8))

        # sorting and fetching indexes for each models based on accuracy
        for i in range(8):
                # sorting and fetching indexes
                p = sorted(range(len(acc[i])), key=lambda k: acc[i][k])
                #print(p)  showing indices of sorted list
                ind[0][i]= p[-1]

        #print(ind[0])

        ind[0][0]

        u = 1

        NeuralNetworkTrain(X_train,y_train,X_test,y_test,ind[0][0],u)
        Logistic_Regression(X_train,y_train,X_test,y_test,ind[0][1],u)
        Naive_Bayes(X_train,y_train,X_test,y_test,ind[0][2],u)
        Stochastic_Gradient_Descent(X_train,y_train,X_test,y_test,ind[0][3],u)
        K_Nearest_Neighbours(X_train,y_train,X_test,y_test,ind[0][4],u)
        Decision_Tree(X_train,y_train,X_test,y_test,ind[0][5],u)
        Random_Forest(X_train,y_train,X_test,y_test,ind[0][6],u)
        Support_Vector_Machine(X_train,y_train,X_test,y_test,ind[0][7],u)











