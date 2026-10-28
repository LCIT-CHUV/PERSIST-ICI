import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

def cluster_kmeans(X,k=2,seed=20260916):
    labels=KMeans(n_clusters=k,n_init=50,random_state=seed).fit_predict(X); return labels,float(silhouette_score(X,labels))
def cluster_ward(X,k=2):
    labels=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(X); return labels,float(silhouette_score(X,labels))
def pca_scores(X,n_components=None):
    n_components=n_components or min(X.shape); p=PCA(n_components=n_components).fit(X); return p.transform(X),p.explained_variance_ratio_
def bootstrap_ari(X,labels,k=2,n=300,seed=20260916):
    rng=np.random.default_rng(seed); vals=[]; N=len(X)
    for _ in range(n):
        idx=rng.integers(0,N,N); km=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(X[idx]); pred=km.predict(X); vals.append(adjusted_rand_score(labels,pred))
    return float(np.mean(vals))
