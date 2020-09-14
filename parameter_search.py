from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_elbow(x_Data, k_max = 10):
    '''
	Computes several KMEANS clustering algorithm over data and plots distortion
	graph to evaluate the optimum K value using the Elbow method.
    '''

    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_jobs=-1).fit(df_X_std)
        kmeanModel.fit(df_X_std)
        distortions.append(sum(np.min(cdist(df_X_std, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df_X_std.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('K vs Distortion value (Elbow method)')
    plt.show()
