
# coding: utf-8

# In[4]:

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import os

import pickle
import multiprocessing

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
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
from sklearn.linear_model import SGDClassifier

#Evalaluation
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV

#Performance

#os.chdir(r"D:/My Computer/")
#os.getcwd()



# ## Gaussian

def buildmodels():

    #Titanic
    path = r"C:/Users/Nicol/Google Drive/Learning/Jupyter/Titanic"
    #train_df = pd.read_csv(open(os.path.join(path, "clean_train.csv"), "r")) 
    #test_df = pd.read_csv(open(os.path.join(path, "clean_test.csv"), "r"))

    train_df = pd.read_csv(open(os.path.join(path, "clean_train2.csv"), "r")) 
    test_df = pd.read_csv(open(os.path.join(path, "clean_test2.csv"), "r")) 

    X = train_df.drop(["Survived"] , axis=1)
    y = train_df["Survived"]

    oosample  = test_df.drop(["PassengerId"] , axis=1).copy()
    print(X.shape, y.shape, oosample.shape)

    results=[]
    def save(model, modelname):
        model.fit(X, y)
        submission = model.predict(oosample)
        df = pd.DataFrame({'PassengerId':test_df.PassengerId, 
                               'Survived':submission})
        df.to_csv((os.path.join(path,("submissions/{}.csv".format(modelname)))),header=True,index=False)
        
        # CV and Save Scores
        trainingscore = (grid.best_score_*100)
        results.append([(trainingscore),("{}".format(modelname)), grid.best_estimator_])
        print(trainingscore)
        print(grid.best_params_)

        with open((os.path.join(path,(r"Pickle/{}.pickle".format(modelname)))), 'wb') as f: pickle.dump(model, f)
            
    def norm_save(model, modelname):
        model.fit(X, y)
        submission = model.predict(oosample)
        df = pd.DataFrame({'PassengerId':test_df.PassengerId, 
                               'Survived':submission})
        df.to_csv((os.path.join(path,("submissions/{}.csv".format(modelname)))),header=True,index=False)
        with open((os.path.join(path,(r"Pickle/{}.pickle".format(modelname)))), 'wb') as f: pickle.dump(model, f)

    def ensembling(model, modelname):
        model.fit(X, y)
        submission = model.predict(oosample)
        df = pd.DataFrame({'PassengerId':test_df.PassengerId, 
                               'Survived':submission})
        print(len(df))
        df.to_csv((os.path.join(path,(r"submissions/{}.csv".format(modelname)))),header=True,index=False)
        with open((os.path.join(path,(r"Pickle/{}.pickle".format(modelname)))), 'wb') as f: pickle.dump(model, f)


    # In[7]:

    print(y.value_counts(normalize=True))


    # In[8]:

    # use train/test split with different random_state values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2017)
    X_train.shape, y_train.shape, X_test.shape, y_test.shape

    # Stratified Cross Validation
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)

    # Normal CV
    #cv =5


    # In[5]:

    print(X.info())
    print(y.head())
    print(oosample.info())

    model = GaussianNB()

    score = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(score.mean())
    norm_save(GaussianNB(), "Gaussian")


    # ## Logistic

    model= LogisticRegression()
    score = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(score.mean())
    norm_save(LogisticRegression(), "Logistic_Regression")


    # # Non-Parametric

    # ## Bagging

    tree = DecisionTreeClassifier()
    bag = BaggingClassifier(tree, n_estimators=300, max_samples=0.8,
                            random_state=1)

    print(cross_val_score(bag, X, y, cv=10, scoring='accuracy').mean()*100)


    param_grid ={'n_estimators': np.arange(20, 500, 25)}

    tree = DecisionTreeClassifier()
    #bag = BaggingClassifier(tree)

    grid = GridSearchCV(BaggingClassifier(tree),
                        param_grid, cv=cv, scoring='accuracy',
                        verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "Bagger_ensemble")


    # ## Random Forest

    model = RandomForestClassifier(n_estimators=500)
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))


    param_grid ={'max_depth': np.arange(6, 11, 1),
                 'n_estimators':np.arange(350, 450, 25),
                 'max_features':np.arange(0.5,.81, 0.05),
                'max_leaf_nodes':np.arange(6, 10, 1)}
    #param_grid ={'n_estimators':[200]}

    from sklearn import feature_selection

    #model = feature_selection.RFE(RandomForestClassifier())
    model= RandomForestClassifier()

    grid = GridSearchCV(model,
                        param_grid, cv=cv,
                        scoring='accuracy',
                        verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "Random_Forest")


    # ## AdaBoostClassifier

    param_grid ={'n_estimators':np.arange(50, 301, 25),
                'learning_rate':np.arange(.1, 4, .5)}

    grid = GridSearchCV(AdaBoostClassifier(),
                        param_grid,cv=cv, scoring='accuracy',
                        verbose=1)

    grid.fit(X, y);
    save(grid.best_estimator_, "AdaBoost_Ensemble")


    # ## Gradient Boosting Classifier
    param_grid ={'n_estimators':np.arange(100, 301, 25),
                'loss': ['deviance', 'exponential'],
                'learning_rate':np.arange(0.01, 0.32,.05),
                'max_depth': np.arange(2, 4.1, .5)}

    grid = GridSearchCV(GradientBoostingClassifier(),
                        param_grid,cv=cv,
                        scoring='accuracy',
                        verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "Gradient_Boosting")

    # ## KNN
    param_grid ={'n_neighbors': np.arange(1,21,1),
                'weights':['uniform','distance']
                }

    grid = GridSearchCV(KNeighborsClassifier(),
                        param_grid,cv=cv, scoring='accuracy',
                        verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "KNN")


    # # Discriminative Classification
    # ### Stochastic Gradient Descent

    param_grid ={'loss':["hinge","log","modified_huber","squared_hinge","epsilon_insensitive","squared_epsilon_insensitive"]
                }

    grid = GridSearchCV(SGDClassifier(),
                        param_grid,cv=cv, scoring='accuracy',
                        verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "StochasticGradientDescent")


    # ## Support Vector Classifier

    model = svm.LinearSVC()
    #Fit Model
    scores= cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(scores.mean()*100)

    #submit(svm.LinearSVC(), name="80linear_svc.csv")


    # ### RBF SVC

    param_grid = [
      {'C': np.arange(25,176,5), 'gamma': np.logspace(1, -4, 10), 'kernel': ['rbf'],"probability" : [True]}
     ]

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(svm.SVC(),
                        param_grid, cv=cv,
                        scoring='accuracy', verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "SVCrbf")


    # ### Linear SVC

    param_grid = {'C': [1,10],'kernel':['linear'], "probability" : [True]}

    grid = GridSearchCV(svm.SVC(),
                        param_grid, cv=cv,
                        scoring='accuracy', verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "SVCLinear")


    # ## PCA + SVC Pipeline

    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline

    pca = PCA(n_components=5, whiten=True, random_state=42, svd_solver='randomized')
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    # pipeline!

    param_grid = [{'svc__C': np.logspace(-2, 3, 6),
                  'svc__gamma': np.logspace(1, -7, 10)},
                  {'svc__C': [0.001, 0.01, 0.1, 1, 10, 50, 100, 150, 1000, 1500], 'svc__kernel': ['linear']}]
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=cv, verbose=1)

    grid.fit(X, y)
    save(grid.best_estimator_, "PCA_SVC")



    results


if __name__ == '__main__':
    p = multiprocessing.Process(target=buildmodels)
    p.start()