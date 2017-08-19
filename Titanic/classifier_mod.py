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


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


open_file = open("Titanic/Pickle/Ada_Boost_Ensemble.pickle", "rb")
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

voted_classifier = VoteClassifier(
                                  Ada_Boost_Ensemble,
                                  Bagger_ensemble,
                                  Gradient_Boosting,
                                  KNN,
                                  PCA_SVC,
                                  SVCdouble_kernel,
                                  SVC2)




def ensembler(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)