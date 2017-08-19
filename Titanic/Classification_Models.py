
# coding: utf-8

# In[1]:

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os

import pickle

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Evalaluation
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV


#Titanic
path = r"C:/Users/Nicol/Google Drive/Learning/Jupyter/Titanic"
train_df = pd.read_csv(open(os.path.join(path, "clean_train.csv"), "r")) 
test_df = pd.read_csv(open(os.path.join(path, "clean_test.csv"), "r")) 

X = train_df.drop(["Survived"] , axis=1)
y = train_df["Survived"]

oosample  = test_df.drop(["PassengerId"] , axis=1).copy()
print(X.shape, y.shape, oosample.shape)



open_file = open("Titanic/Pickle/AdaBoost_Ensemble.pickle", "rb")
Ada_Boost_Ensemble = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/Bagger_ensemble.pickle", "rb")
Bagger_ensemble = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/Gradient_Boosting.pickle", "rb")
Gradient_Boosting = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/KNN.pickle", "rb")
KNN = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/PCA_SVC.pickle", "rb")
PCA_SVC = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/Random_Forest.pickle", "rb")
Random_Forest = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/SVCdouble_kernel.pickle", "rb")
SVCdouble_kernel = pickle.load(open_file)
open_file.close()

open_file = open("Titanic/Pickle/SVC2.pickle", "rb")
SVC2 = pickle.load(open_file)
open_file.close()


# In[55]:

from sklearn.ensemble import VotingClassifier
get_ipython().magic('pinfo VotingClassifier')



est=[
    ('ada',Ada_Boost_Ensemble),
    ('bagger',Bagger_ensemble),
    ('gradientboost',Gradient_Boosting),
    ('knn',KNN),
    ('pca_svc',PCA_SVC),
    ('svc1',SVCdouble_kernel),
    ('svc2',SVC2)
]

ensemble = VotingClassifier(estimators=est,
                            #weights=[2,2],
                            #flatten_transform=True,
                            voting='soft')
ensemble.fit(X,y)


# In[100]:

print(ensemble.predict_proba(X))


# In[101]:

ensemble.score(X,y)


# In[13]:

def ensembling(model, modelname):
    model.fit(X, y)
    submission = model.predict(oosample)
    df = pd.DataFrame({'PassengerId':test_df.PassengerId, 
                           'Survived':submission})
    print(len(df))
    df.to_csv(("Titanic/submissions/{}.csv".format(modelname)),header=True,index=False)


# In[14]:

ensembling(ensemble, "ensemble2")
