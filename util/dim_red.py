from matplotlib.mlab import PCA as mlabPCA
from matplotlib import pyplot as plt
import numpy as np
from util.cluster import My_KMeans


class MyPCA:

    def __init__(self, feature_matrix, cluster_labels, means):
        self.feature_matrix = feature_matrix
        self.cluster_means = means
        self.cluster_labels = cluster_labels

    def run(self):
        mlab_pca = mlabPCA(self.feature_matrix)

        project_matrix = mlab_pca.Wt
        project_means = np.matmul(self.cluster_means, project_matrix)

        # collect userIdices for each cluster
        cluster_users = {}
        for userIdx, clusterIdx in self.cluster_labels.items():
            if clusterIdx not in cluster_users:
                cluster_users[clusterIdx] = []
            cluster_users[clusterIdx].append(userIdx)

        colors = ['b', 'c', 'g', 'k', 'm', 'r', 'y']
        dots = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's' 'p', '*', 'h', 'H', 'd', '|', '_', '+', 'x']

        dot_count = 0
        color_count = 0
        cluster_plot_conf = {}
        for clusterIdx in set(self.cluster_labels.values()):
            cluster_plot_conf[clusterIdx] = [dots[dot_count], colors[color_count]]
            color_count += 1
            if color_count == len(colors):
                dot_count += 1
                color_count = 0

        # draw plot
        # plt.plot(mlab_pca.Y[0:20,0],mlab_pca.Y[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
        ax = plt.subplot(111, projection='3d')

        for clusterIdx, userIdices in cluster_users.items():
            for userIdx in userIdices:
                ax.scatter(mlab_pca.Y[userIdx, 0], mlab_pca.Y[userIdx, 1], mlab_pca.Y[userIdx, 2], cluster_plot_conf[clusterIdx][0], color=cluster_plot_conf[clusterIdx][1])

        ax.scatter(project_means[:, 0], project_means[:, 1], project_means[:, 2], 'x', color='r')

        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')

        plt.show()

if __name__ == '__main__':

    feature_matrix = np.loadtxt('../save_model/' + 'ml-100k' + '-' + 'BPRRecommender' + '-user_embed.txt')
    cluster = My_KMeans(feature_matrix, num_cluster=10)
    cluster.build_cluster()

    mypca = MyPCA(feature_matrix, cluster.u_cluster_labels, cluster.means)
    mypca.run()



