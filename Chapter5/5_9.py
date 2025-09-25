import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier

np.random.seed(42)
col1 = np.random.uniform(0, 35, 1000)
col2 = np.random.uniform(0, 17, 1000)
X = np.column_stack([col1,col2])
X = X.round(2)
y = (X[:, 0] + X[:, 1] > 41).astype(int)
X_train = X[:800]
y_train = y[:800]
X_test = X[800:]
y_test= y[800:]
svm_clf = make_pipeline(StandardScaler(),
                        LinearSVC(C=1, dual=True, random_state=42))
svm_clf.fit(X_train, y_train)

print("LinearSVC", svm_clf.score(X_test,y_test))

parameters = {'kernel': ('linear','rbf'), 'gamma':[0.001,10], 'C':[0.001,10]}

svc_clf = GridSearchCV(estimator=SVC(), param_grid=parameters)

svc_clf.fit(X_train, y_train)

# print(svc_clf.best_params_)

print("SVC:",svc_clf.best_estimator_.score(X_test,y_test))


sgd_params = {'loss': ('hinge','log_loss'), 'penalty': ('l1','l2','elasticnet'), 'alpha': [0.0001,0.001,0.01,0.1, 1] }
sgd_clf = GridSearchCV(estimator=SGDClassifier(),param_grid=sgd_params)
sgd_clf.fit(X_train, y_train)

# print(sgd_clf.best_params_)
print("SGDC: ", sgd_clf.best_estimator_.score(X_test,y_test))


