import pandas as pd
from pathlib import Path


titanic_data = pd.read_csv(Path("./titanic/train.csv"))
titanic_to_run = pd.read_csv(Path("./titanic/test.csv"))
# cchwilowe, potrzebne do wspolczynnika korelacji pearsona
titanic_data.drop(['Name', 'Sex','Ticket', 'Cabin', 'Embarked'], inplace=True, axis=1)
# Dziele zbior na testowy i treningowy bo test.csv nie ma etykiet
titanic_test = titanic_data[713:]
titanic_train = titanic_data[:713]


# rozdzielam y i X
titanic_train_y = titanic_train.loc[:,['Survived']]
titanic_test_y = titanic_test.loc[:,['Survived']]
titanic_train_X = titanic_train.drop(['Survived'], axis=1)
titanic_test_X = titanic_test.drop(['Survived'], axis=1)

corr_matrix = titanic_train.corr()

print(corr_matrix["Survived"].sort_values(ascending=False))
# TODO: obrobka danych i przerobienie wszystkiego na integery