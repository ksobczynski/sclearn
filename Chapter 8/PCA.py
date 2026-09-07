from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)

X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
# print(X_train, y_train)

pca = PCA()
pca.fit(X_train)
# print(np.round(pca.explained_variance_ratio_,4))

cumsum = np.cumsum(pca.explained_variance_ratio_)

# print(cumsum)

d = np.argmax(cumsum >= 0.95) + 1
# print("Odpowiednia hiperpłaszczyzna ma " + str(d) + " wymiarów.")

# Lepsza Alternatywa:

pca2 = PCA(n_components=0.95)

X_redu = pca2.fit_transform(X_train)

# print(pca2.n_components_)

# Wizualizacja - do pomocy wyboru odpowiedniego współczynnika zachowania wariancji

fig, ax = plt.subplots()

ax.plot(cumsum)
ax.set(xlabel='d - hyperplane dimension', ylabel = 'variance accuracy')
ax.grid()
fig.savefig("PCA_hyperplane_dimension.png")

# plt.show()

# Przykładowy pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

clf = make_pipeline(PCA(random_state=42), RandomForestClassifier(random_state=42))

param_distrib = {
    "pca__n_components": np.arange(10,80),
    "randomforestclassifier__n_estimators": np.arange(50,500)
}
rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3,random_state=42)
rnd_search.fit(X_train[:1000], y_train[:1000])
# print(rnd_search.best_params_, rnd_search.best_estimator_)

# IPCA - Incremental PCA

from sklearn.decomposition import IncrementalPCA
n_batches = 100

ipca = IncrementalPCA(n_components=154) # optymalna ilosc wyliczona

for X_batch in np.array_split(X_train, n_batches):
    ipca.partial_fit(X_batch)

X_reduced = ipca.transform(X_train)


# Zoptymalizowany ipca - podzial duzej tablicy

filename = "my_mnist.mmap"

X_mmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mmap[:] = X_train
X_mmap.flush()

X_mmap = np.memmap(filename, dtype = "float32", mode="readonly").reshape(-1,784)
batch_size = X_mmap.shape[0] # n batches

ipca = IncrementalPCA(n_components=154, batch_size = batch_size)
ipca.fit(X_mmap)

# Idea  - mapując array z numpy jako plik uywając mmap zapobiegamy zapchaniu RAM-u. Skoro działamy w mmapie mamy page cache i w miare nie mamy narzutu na kernel-space



