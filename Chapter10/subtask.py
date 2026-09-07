# Trying out MPL classifier on Iris dataset

from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris()

mlpc = MLPClassifier(hidden_layer_sizes=[10], max_iter=10_000, )
X_train_full, X_test, y_train_full, y_test = train_test_split(
    iris.data, iris.target, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
mlp_clf = MLPClassifier(hidden_layer_sizes=[7],max_iter=10_000)
pipeline = make_pipeline(StandardScaler(), mlp_clf)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_valid, y_valid)
print("Accuracy - ", accuracy)