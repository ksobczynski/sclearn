from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV

housing = fetch_california_housing() 
X = housing.data
y = housing.target

# print(X)
# print(y)


X_train,X_test, y_train, y_test = train_test_split(X,y,train_size=0.8)

X_hp_tuning = X_train[:2000]
y_hp_tuning = y_train[:2000]
scaler = StandardScaler()
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('classifier',SVR())
])
params = {
    'classifier__C': [0.001,0.01,0.1,1,10,100],
    'classifier__degree': [0,1,2,3],
    'classifier__epsilon': [0.001,0.01,0.1,1]
}

# print(type(X_train))
# print(y_hp_tuning)
svr_gs = GridSearchCV(pipeline, param_grid=params,scoring='neg_mean_squared_error')

svr_gs.fit(X_hp_tuning,y_hp_tuning)

print("Parametry: ", svr_gs.best_params_)
best_pipeline = svr_gs.best_estimator_

best_pipeline.fit(X_train, y_train)



pred_y = best_pipeline.predict(X_test[0].reshape(1,-1))

print("prawdziwy:",  y_test[0], "pred: ", pred_y)


y_pred_full = best_pipeline.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred_full):.4f}")
print(f"R²: {r2_score(y_test, y_pred_full):.4f}")