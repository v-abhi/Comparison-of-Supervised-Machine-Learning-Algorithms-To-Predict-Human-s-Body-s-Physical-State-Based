#!/usr/bin/env python
# coding: utf-8




from sklearn.externals import joblib
import numpy as np
import sys




# Loading generated model 

mod_decision_tree = joblib.load('dt.pkl')
mod_knn = joblib.load('knn.pkl')
mod_logistic_regression = joblib.load('log_reg.pkl')
mod_naive_bayes = joblib.load('Gaussian.pkl')
mod_neural_network = joblib.load('n_n.pkl')
mod_random_forest = joblib.load('rf.pkl')
mod_sgd = joblib.load('sgd.pkl')
mod_svm = joblib.load('svm.pkl')





Activity_list1 = ['stand',
                 'sit',
                 'walk',
                 'stairsup',
                 'stairsdown',
                 ]
                 
Activity_list = ['a','b','c']




# taking input from command line arguement
X_ip = np.array([list(map(float,str(sys.argv[1]).split(',')))])
#print(len(X_ip[0]))
#print(X_ip)


# Use the loaded model to make predictions 
while(len(X_ip[0])) == 4:
        if X_ip[0][-1] == 1.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_decision_tree.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 2.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_knn.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 3.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_logistic_regression.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 4.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_naive_bayes.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 5.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_naive_bayes.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 6.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_random_forest.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 7.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_sgd.predict(b)[0]-1)])
            break;
        elif X_ip[0][-1] == 8.0:
            b = X_ip[0][:-1].copy()
            #print(b)
            b = np.reshape(b, (-1, len(b)))
            print(Activity_list1[int(mod_svm.predict(b)[0]-1)])
            break;
        else:
            print("error\n")
            break;





