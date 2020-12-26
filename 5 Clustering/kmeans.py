import os
import csv
import numpy as np
import urllib.request
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs

def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)

if __name__ == '__main__':
    n_cluster = np.random.randint(2, 11)
    centers = [(np.random.random() * 10 - 5, np.random.random() * 10 - 5) for _ in range(n_cluster)]
    print(n_cluster, centers)
    x, y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)
    print(x.shape)
    print(y.shape)

    for i in range(n_cluster - 2, n_cluster + 3):
        estimator = KMeans(n_clusters=i).fit(x)
        labels = estimator.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels))

        print('Estimated number of clusters: %d' % n_clusters_)
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(x, labels))
        print("BIC: %0.3f" % compute_bic(estimator, x))
        print()
        
        if i == n_cluster:
            # #############################################################################
            # Plot result
            import matplotlib.pyplot as plt

            # Black removed and is used for noise instead.
            unique_labels = set(labels)
            colors = [plt.cm.Spectral(each)
                    for each in np.linspace(0, 1, len(unique_labels))]
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = [0, 0, 0, 1]

                class_member_mask = (labels == k)

                xy = x[class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                        markeredgecolor='k', markersize=6)

                # xy = X[class_member_mask & ~core_samples_mask]
                # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                #         markeredgecolor='k', markersize=6)

            plt.title('Estimated number of clusters: %d' % n_clusters_)
            plt.savefig('kmeans.png')
