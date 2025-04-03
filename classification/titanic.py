import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error


titanic_data = pd.read_csv(Path("./titanic/train.csv"))
titanic_to_run = pd.read_csv(Path("./titanic/test.csv"))
# cchwilowe, potrzebne do wspolczynnika korelacji pearsona
titanic_data.drop(['Name','Ticket', 'Cabin'], inplace=True, axis=1)
# Dziele zbior na testowy i treningowy bo test.csv nie ma etykiet
titanic_test = titanic_data[713:]
titanic_train = titanic_data[:713]

# rozdzielam y i X
titanic_train_y = titanic_train.loc[:,['Survived']]
titanic_test_y = titanic_test.loc[:,['Survived']]
titanic_train_X = titanic_train.drop(['Survived'], axis=1)
titanic_test_X = titanic_test.drop(['Survived'], axis=1)


gender = titanic_train_X[["Sex"]]
# print(gender.head(10))
# print(gender)
# biore 2 oh encodery do podmianki string values
ohencoder_sex = OneHotEncoder()
ohencoder_emb = OneHotEncoder()

#uzywam encoderow
encoded_sex = ohencoder_sex.fit_transform(titanic_train_X[['Sex']])
encoded_embarked = ohencoder_emb.fit_transform(titanic_train_X[['Embarked']])

# print(ohencoder_emb.categories_)

#porzucam i czyszcze dataframe
titanic_train_X[ohencoder_sex.categories_[0]] = encoded_sex.toarray()
titanic_train_X[ohencoder_emb.categories_[0]] = encoded_embarked.toarray()
titanic_train_X.drop(['Sex', 'Embarked'], axis=1, inplace=True)
titanic_train_X.drop(titanic_train_X.columns[11], axis=1, inplace=True)
# print(titanic_train_X.columns)

imputer = SimpleImputer(strategy="median")
only_nr_age = titanic_train_X.select_dtypes(include=[np.number])
imputer.fit(only_nr_age)
# print(imputer.statistics_)
mediana = titanic_train_X["Age"].median()
titanic_train_X["Age"].fillna(mediana, inplace=True)
# print(titanic_train_X.columns)
# print(titanic_train_X.head(10))
# print(titanic_train_y)
# TODO: obrobka danych i przerobienie wszystkiego na integery

param_distro = {'n_neighbors': randint(low=2, high=10),
                'weights': ['uniform', 'distance'],
                'algorithm' : ['auto' , 'ball_tree' , 'kd_tree' ,'brute']
                }
knn_clasifier = KNeighborsClassifier(n_neighbors= 5,weights ='uniform', # it can be distance
                                          algorithm='auto')
series_y_train = titanic_train_y['Survived'].to_numpy()
# print(series_y_train)
CrossValidate = cross_validate(knn_clasifier,titanic_train_X,series_y_train,cv=5,return_train_score = True)
print('Train Score Value : ', CrossValidate['train_score'])
print('Test Score Value : ', CrossValidate['test_score'])
print('Fit Time : ', CrossValidate['fit_time'])
print('Score Time : ', CrossValidate['score_time']) 
'''
Wyniki dosc srednie:

Train Score Value :  [0.76315789 0.74912281 0.75263158 0.74430823 0.73204904]
Test Score Value :  [0.60839161 0.46153846 0.48951049 0.43661972 0.67605634]
Fit Time :  [0.00188804 0.0006001  0.00054002 0.00050473 0.00050187]
Score Time :  [0.0027132  0.00195193 0.00199294 0.0018692  0.00224805]
'''
print('\n')
clf = RandomizedSearchCV(knn_clasifier, param_distro, cv=5 , n_iter = 15)
clf.fit(titanic_train_X, series_y_train)
print('score : ' , clf.best_score_)
print('params : ' , clf.best_params_)
print('best : ' , clf.best_estimator_)

'''
score :  0.5498965822909485
params :  {'algorithm': 'ball_tree', 'n_neighbors': 6, 'weights': 'uniform'}
best :  KNeighborsClassifier(algorithm='ball_tree', n_neighbors=6)
kosteks@Mac classification % 
'''
# print([x/100 for x in range(100)])
param_distro = {'loss': ['log_loss', 'exponential'],
                'learning_rate': uniform(loc=0.01,scale=0.99),
                'n_estimators' : randint(50,150),
                'subsample' : uniform(loc=0.01,scale=0.99),
                'criterion' : ['friedman_mse','squared_error'],
                'min_samples_split' : randint(2,10),
                'min_samples_leaf' :  randint(1,10),
                'min_weight_fraction_leaf' : uniform(loc=0.0, scale=0.5),
                'max_depth' : randint(1,10),
                'min_impurity_decrease' : uniform(loc=0.0, scale=5.0)
                }

print("GB Classifier: ")

gbclassifier = GradientBoostingClassifier()

CrossValidate = cross_validate(gbclassifier,titanic_train_X,series_y_train,cv=5,return_train_score = True)
print('Train Score Value : ', CrossValidate['train_score'])
print('Test Score Value : ', CrossValidate['test_score'])
print('Fit Time : ', CrossValidate['fit_time'])
print('Score Time : ', CrossValidate['score_time']) 

clf = RandomizedSearchCV(gbclassifier, param_distro, cv=5 , n_iter = 50)
clf.fit(titanic_train_X, series_y_train)
print('score : ' , clf.best_score_)
print('params : ' , clf.best_params_)
print('best : ' , clf.best_estimator_)
'''
score :  0.7896680784004727
params :  {'criterion': 'friedman_mse', 'learning_rate': np.float64(0.7046413926076784), 'loss': 'exponential', 'max_depth': 7, 'min_impurity_decrease': np.float64(2.057871015953816), 'min_samples_leaf': 1, 'min_samples_split': 9, 'min_weight_fraction_leaf': np.float64(0.011715877390825802), 'n_estimators': 78, 'subsample': np.float64(0.8699259821144918)}
best :  GradientBoostingClassifier(learning_rate=np.float64(0.7046413926076784),
                           loss='exponential', max_depth=7,
                           min_impurity_decrease=np.float64(2.057871015953816),
                           min_samples_split=9,
                           min_weight_fraction_leaf=np.float64(0.011715877390825802),
                           n_estimators=78,
                           subsample=np.float64(0.8699259821144918))
'''

print("\nTesting on test dataset\n ")

# best_model = GradientBoostingClassifier(learning_rate=np.float64(0.7046413926076784),
#                            loss='exponential', max_depth=7,
#                            min_impurity_decrease=np.float64(2.057871015953816),
#                            min_samples_split=9,
#                            min_weight_fraction_leaf=np.float64(0.011715877390825802),
#                            n_estimators=78,
#                            subsample=np.float64(0.8699259821144918))
preds = clf.best_estimator_.predict(titanic_test_X)
final_rmse = mean_squared_error(titanic_test_y, preds)
print(final_rmse)
#TODO kod nie dziala, trzeba jeszcze zrobic pipeline dla titanic_test_X zeby wygladal tak jak zestaw treningowy i moze byc okej, 80 procent juz mnie zadowala. posprzataj potem kod jeszcze