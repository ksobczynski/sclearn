from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)

X,y = mnist.data, mnist.target
def plot_digit(image_data):
    image = image_data.reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
# plt.show()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# parameters = {'weights':['uniform', 'distance'], 'n_neighbors':[x+1 for x in range(10)]}
# grid_search = GridSearchCV(knc,parameters, cv=5)
# grid_search.fit(X_train, y_train)
# print("Best estimator:", grid_search.best_estimator_)
# print("Best parameters:", grid_search.best_params_)
# print("Best score:", grid_search.best_score_)

# Best estimator: KNeighborsClassifier(n_neighbors=4, weights='distance')
# Best parameters: {'n_neighbors': 4, 'weights': 'distance'}
# Best score: 0.9716166666666666
# knc.fit()
knc = KNeighborsClassifier(weights='distance', n_neighbors=4)
knc.fit(X_train, y_train)
# y_train_pred = cross_val_score(knc, X_train, y_train, cv=3, scoring="accuracy")
# print(y_train_pred)


test_preds = knc.predict(X_test)
final_precision = accuracy_score(y_test, test_preds)
print("Precision: ", final_precision)
# print(X_train[0])
