from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784', as_frame=False)

X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

X_sample, y_sample = X_train[:5000], y_train[:5000]


tsne = TSNE(n_components=2)



X_transformed = tsne.fit_transform(X_sample,y_sample)

# print(X_transformed)

plt.figure(figsize=(13,10))
plt.scatter(X_transformed[:,0], X_transformed[:,1],c=y_sample.astype(np.int8), cmap="jet", alpha=0.5)


plt.axis('off')
plt.colorbar()
plt.show()