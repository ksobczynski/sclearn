from sklearn.datasets import load_wine
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


wine = load_wine(as_frame=True)
print(wine.data.columns)
y = wine.target.to_numpy()
# print(y)
X = wine.data.to_numpy()
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

parameters = {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[1, 10]}

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('classifier',SVC())
])
svc = GridSearchCV(pipeline, parameters,cv=5,scoring='accuracy')
svc.fit(X_train, y_train)

svc.fit(X_train,y_train)

print(svc.best_estimator_.score(X_test,y_test))


# ex = X_test[1].reshape(1,-1)

# print(svc.predict(ex))
# print(y_test[1])


