# Rzutowanie losowe - teoria Johnsona - Lindenstraussa


from sklearn.random_projection import johnson_lindenstrauss_min_dim

m, eps = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=eps)
print(d)

# losowanie macierzy lsoowej P
import numpy as np

n = 20_000
np.random.seed(42)
P = np.random.randn(d,n) / np.sqrt(d)

X = np.random.randn(m,n) # sztuczny zestaw danych
X_reduced = X @ P.T

# wbudowane w sklearn

from sklearn.random_projection import GaussianRandomProjection

grp = GaussianRandomProjection(eps = eps, random_state=42)
X_reduced2 = grp.fit_transform(X)



