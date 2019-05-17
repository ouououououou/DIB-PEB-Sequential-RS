import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib.cm as cm

class cluster_class:
    def __init__(self, feature_matrix):
        self.feature_matrix = feature_matrix
        self.u_cluster_labels = {}
        self.num_clusters = 0

class My_DBSCAN(cluster_class):

    def __init__(self, feature_matrix):
        super(DBSCAN, self).__init__(feature_matrix)
        self.u_cluster_labels = {}
        self.num_clusters = 0


    def build_cluster(self):
        # # #############################################################################
        # # Generate sample data
        # centers = [[1, 1], [-1, -1], [1, -1]]
        # X, y = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
        #                             random_state=0)
        #
        # self.sim_matrix = StandardScaler().fit_transform(self.sim_matrix)
        #
        # # #############################################################################
        # Compute DBSCAN
        print('building cluster')
        y_pred = DBSCAN(eps=0.1, min_samples=1).fit_predict(self.feature_matrix)

        for userIdx in range(self.feature_matrix.shape[0]):
            cluster_idx = y_pred[userIdx]
            self.u_cluster_labels[userIdx] = cluster_idx

        # Number of clusters in labels, ignoring noise if present.
        self.num_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

        print('Estimated number of clusters: %d' % self.num_clusters)
        # print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, y_pred))
        # print("Completeness: %0.3f" % metrics.completeness_score(y, y_pred))
        # print("V-measure: %0.3f" % metrics.v_measure_score(y, y_pred))
        # print("Adjusted Rand Index: %0.3f"
        #       % metrics.adjusted_rand_score(y, y_pred))
        # print("Adjusted Mutual Information: %0.3f"
        #       % metrics.adjusted_mutual_info_score(y, y_pred))
        # print("Silhouette Coefficient: %0.3f"
        #       % metrics.silhouette_score(X, y_pred))

        # # #############################################################################
        # # Plot result
        # import matplotlib.pyplot as plt
        #
        # # Black removed and is used for noise instead.
        # unique_labels = set(y_pred)
        # colors = [plt.cm.Spectral(each)
        #           for each in np.linspace(0, 1, len(unique_labels))]
        # for k, col in zip(unique_labels, colors):
        #     if k == -1:
        #         # Black used for noise.
        #         col = [0, 0, 0, 1]
        #
        #     class_member_mask = (y_pred == k)
        #
        #     xy = X[class_member_mask & core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=14)
        #
        #     xy = X[class_member_mask & ~core_samples_mask]
        #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
        #              markeredgecolor='k', markersize=6)
        #
        # plt.title('Estimated number of clusters: %d' % n_clusters_)
        # plt.show()

class My_KMeans(cluster_class):

    def __init__(self, feature_matrix, num_cluster):
        super(My_KMeans, self).__init__(feature_matrix)
        self.u_cluster_labels = {}
        self.num_clusters = num_cluster
        self.means = None


    def build_cluster(self):


        kmeans = KMeans(n_clusters=self.num_clusters, max_iter=300, n_init=40, \
                            init='k-means++', n_jobs=-1)

        kmeans.fit(self.feature_matrix)
        y_pred = kmeans.labels_  # 获取聚类标签
        self.means = kmeans.cluster_centers_  # 获取聚类中心

        for userIdx in range(self.feature_matrix.shape[0]):
            cluster_idx = y_pred[userIdx]
            self.u_cluster_labels[userIdx] = cluster_idx

        # Number of clusters in labels, ignoring noise if present.
        self.num_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

        # print('Estimated number of clusters: %d' % self.num_clusters)
        #
        # # For each number of clusters, perform Silhouette analysis and visualize the results.
        # n_clusters = self.num_clusters
        #
        # # Compute the Silhouette Coefficient for each sample.
        # s = metrics.silhouette_samples(self.feature_matrix, y_pred)
        #
        # # Compute the mean Silhouette Coefficient of all data points.
        # s_mean = metrics.silhouette_score(self.feature_matrix, y_pred)
        #
        # # For plot configuration -----------------------------------------------------------------------------------
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.set_size_inches(28, 17)
        #
        # # Configure plot.
        # plt.suptitle('Silhouette analysis for K-Means clustering with n_clusters: {}'.format(n_clusters),
        #              fontsize=14, fontweight='bold')
        #
        # # Configure 1st subplot.
        # ax1.set_title('Silhouette Coefficient for each sample')
        # ax1.set_xlabel("The silhouette coefficient values")
        # ax1.set_ylabel("Cluster label")
        # ax1.set_xlim([-1, 1])
        # ax1.set_ylim([0, len(self.feature_matrix) + (n_clusters + 1) * 10])
        #
        # ax2.set_xlabel("Feature space for the 1st feature")
        # ax2.set_ylabel("Feature space for the 2nd feature")
        #
        # # For 1st subplot ------------------------------------------------------------------------------------------
        #
        # # Plot Silhouette Coefficient for each sample
        # y_lower = 10
        # for i in range(n_clusters):
        #     ith_s = s[y_pred == i]
        #     ith_s.sort()
        #     size_cluster_i = ith_s.shape[0]
        #     y_upper = y_lower + size_cluster_i
        #     color = cm.spectral(float(i) / n_clusters)
        #     ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_s,
        #                       facecolor=color, edgecolor=color, alpha=0.7)
        #     ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        #     y_lower = y_upper + 10
        #
        # # Plot the mean Silhouette Coefficient using red vertical dash line.
        # ax1.axvline(x=s_mean, color="red", linestyle="--")
        #
        # plt.show()


if __name__ == '__main__':
    for cluster_num in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        feature_matrix = np.loadtxt('../save_model/' + 'ml-100k' + '-' + 'BPRRecommender' + '-user_embed.txt')
        cluster = My_KMeans(feature_matrix, num_cluster=cluster_num)
        cluster.build_cluster()

    # np.savetxt('../save_model/' + 'ml-100k' + '-' + 'BPRRecommender' + '-user_embed.txt', cluster.means)

