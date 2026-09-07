from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml
import time

mnist = fetch_openml('mnist_784', as_frame=False)


X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

forest = RandomForestClassifier(n_estimators=75)


start = time.time()


forest.fit(X_train, y_train)
end = time.time()

print("unchanged dimension score: ", forest.score(X_test, y_test), "\ntime: ", end-start)


# reduced dim:

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

clf = make_pipeline(PCA(n_components=0.95), RandomForestClassifier(n_estimators=75))

start = time.time()

clf.fit(X_train, y_train) # tu potok nie pomaga samo przeliczenie PCA drogie
end = time.time()

print("changed dimension score: ", clf.score(X_test, y_test), "\ntime: ", end-start)
