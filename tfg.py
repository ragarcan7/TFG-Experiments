# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 10:46:55 2021

@author: Usuario
"""

import numpy as np
#import keras
import pandas
#import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
from time import time
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold as StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import datasets,metrics,decomposition
#from tensorflow.keras import datasets, layers, models
            
            
# reading the CSV file
csvFile = pandas.read_csv('D:\\UV\\4GII\\TFG\\codigo\\Malware dataset.csv')

datos = csvFile


datos['classification'] = datos.classification.map({'benign':0, 'malware':1})

datos = datos.sample(frac=1).reset_index(drop=True)

h = .02
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ['darkorange', 'c', 'darkblue']

splits = [0.1,0.25,0.5,0.75,0.9]

sns.countplot(datos["classification"])
plt.show()

X = datos.drop(["hash","classification",'vm_truncate_count','shared_vm','exec_vm','nvcsw','maj_flt','utime'],axis=1)
y = datos["classification"]

for split in splits:

    X_tr, X_ts, y_tr, y_ts = train_test_split(X , y, test_size=split, random_state=42)
    
    scaler = StandardScaler()
    
    X_tr = scaler.fit_transform(X_tr)
    X_ts = scaler.transform(X_ts)
    
    tr_time = 0
    
    print('-------Training kNN------------')
    
    t_ini = time()
    Acc_kn = []             
    knn = KNeighborsClassifier(n_neighbors= 10)
    knn.fit(X_tr, y_tr)
    pred_kn = knn.predict(X_ts)
    Acc_kn.append(metrics.accuracy_score(y_ts,pred_kn))
    
    tr_time += time() - t_ini
    
    print('Time: {}'.format(tr_time))
    
    print('Accuarcy KNN:')  
    print(Acc_kn)
    
    Precision = metrics.precision_score(y_ts,pred_kn)
    Recall = metrics.recall_score(y_ts,pred_kn)
    
    print('')
    print ('Precision and Recall: {}, {}'.format(Precision,Recall))
    CF = metrics.confusion_matrix(y_ts,pred_kn,labels=[0,1])
    print('Confusion matrix:\n {}'.format(CF))
    
    
    classifiers = [
        ("MLP",1,MLPClassifier(hidden_layer_sizes=(15,15,15), activation='relu', max_iter=100, alpha=1e-4, solver='adam', tol=1e-4, random_state=1, learning_rate_init=.001)),
    ]
    
    rounds = 5 # Number of repetitions to compute average error
    
    seed = np.random.randint(100)
    
    for name, lws, clf in classifiers:
        print('-------Training %s------------' % name)
        rng = np.random.RandomState(seed)  #to have the same for all classifiers
        yyTr = []
        yyTs = []
        
        tr_time = 0
    
        ssumTr = 0
        ssumTs = 0
        for r in range(rounds):
    
            t_ini = time()
            clf.fit(X_tr, y_tr)
    
            y_pred = clf.predict(X_ts)
            tr_time += time() - t_ini
    
            ssumTr += clf.score(X_tr,y_tr)
            ssumTs += clf.score(X_ts,y_ts)
    
        yyTr.append(ssumTr/rounds)
        yyTs.append(ssumTs/rounds)
    
        print("Average training time after {} rounds: {}".format(rounds,tr_time/rounds))
        print("average accuracy: {}".format(yyTs[-1]))
        
        Precision = metrics.precision_score(y_ts,y_pred,average='weighted')
        Recall = metrics.recall_score(y_ts,y_pred,average='weighted')
        print ('Precision and Recall: {}, {}'.format(Precision,Recall))
        
        CF = metrics.confusion_matrix(y_ts,y_pred,labels=[0,1])
        print('Confusion matrix:\n {}'.format(CF))
