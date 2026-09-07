# LLE - locally linear embedding

from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2,random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10,random_state=42)
X_unrolled = lle.fit_transform(X_swiss)

