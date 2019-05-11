#!/usr/X_ipin/env python
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
                 
Activity_list = ['a','X_ip','c']




# taking input from command line arguement
X_ip = np.array([list(map(float,str(sys.argv[1]).split(',')))])

output = ""

# Use the loaded model to make predictions 



temp = Activity_list1[int(mod_decision_tree.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_knn.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_logistic_regression.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_naive_bayes.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_neural_network.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_random_forest.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_sgd.predict(X_ip)[0]-1)]
output = output+" "+temp


temp = Activity_list1[int(mod_svm.predict(X_ip)[0]-1)]
output = output+" "+temp

print(output)
           





