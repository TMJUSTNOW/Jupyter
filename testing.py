from sklearn.datasets import make_friedman1
from sklearn import feature_selection
from sklearn import preprocessing
from sklearn import pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVR

class myPipe(pipeline.Pipeline):

    def fit(self, X,y):
        """Calls last elements .coef_ method.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------
        """

        super(myPipe, self).fit(X,y)

        self.coef_=self.steps[-1][-1].coef_
        return

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)

est = SVR(kernel="linear")

selector = feature_selection.RFE(est)
std_scaler = preprocessing.StandardScaler()
pipe_params = [('std_scaler', std_scaler),('select', selector), ('clf', est)]

pipe = myPipe(pipe_params)



selector = feature_selection.RFE(pipe)
clf = GridSearchCV(selector, param_grid={'estimator__clf__C': [2, 10]})
clf.fit(X, y)

print(clf.best_estimator_)
print(clf.best_params_)