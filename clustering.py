import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from pandas import plotting as pg
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


def hierarchy_cluster(hotels_data):
    names = hotels_data.iloc[:, 0]
    # countries = hotels_data.iloc[:, 1]
    stars = hotels_data.iloc[:, 2]
    hotels_data = hotels_data.drop(columns=["NOM", "PAYS", "ETOILE"], index=1)  # pop non numeric data
    columns = hotels_data.columns
    # print(hotels_data)
    corr = hotels_data.corr()
    print("correlation matrix : \n", corr)
    # fig, (ax1, ax2) = plt.subplots(2)
    # pg.scatter_matrix(hotels_data)
    # plt.show()
    data_array = hotels_data.to_numpy()
    scaler = StandardScaler()
    scaler.fit(data_array)  # Compute the mean and std to be used for later scaling. Necessary before transform
    fitted_data_array = scaler.transform(data_array)  # fit the data (mean = 0) and scale the data (std = 1)
    print("Mean : ", np.mean(fitted_data_array), "\nStd : ", np.var(fitted_data_array))
    methods = ['single', 'complete', 'average', 'centroid', "ward"]
    Z_single = hierarchy.linkage(fitted_data_array, 'single')
    '''fig, axes = plt.subplots(5)
    dn1 = hierarchy.dendrogram(Z_single, ax=axes[0], color_threshold=0)

    Z_complete = hierarchy.linkage(fitted_data_array, 'complete')
    dn2 = hierarchy.dendrogram(Z_complete, ax=axes[1], color_threshold=0)

    Z_average = hierarchy.linkage(fitted_data_array, 'average')
    dn3 = hierarchy.dendrogram(Z_average, ax=axes[2], color_threshold=0)

    Z_centroid = hierarchy.linkage(fitted_data_array, 'centroid')
    dn4 = hierarchy.dendrogram(Z_centroid, ax=axes[3], color_threshold=0) '''

    Z_ward = hierarchy.linkage(fitted_data_array, 'ward')
    dn5 = hierarchy.dendrogram(Z_ward)
    # plt.show()

    '''for t in range(1, 10):
        labels = hierarchy.fcluster(Z_ward, t, criterion='distance')
        silhouetteScore = silhouette_score(fitted_data_array, labels)
        print("coeff de silhouette :", silhouetteScore, "\nnb clusters : ", labels[np.argmax(labels)])'''

    labels = hierarchy.fcluster(Z_ward, t=7, criterion='distance')
    silhouetteScore = silhouette_score(fitted_data_array, labels)
    print("coeff de silhouette :", silhouetteScore, "\nnb clusters : ", labels[np.argmax(labels)])
    for i in range(0, len(labels) - 1):
        print(names[np.argsort(labels)[i]], labels[np.argsort(labels)[i]])


def kmeans(hotels_data):
    names = hotels_data.iloc[:, 0]
    # countries = hotels_data.iloc[:, 1]
    stars = hotels_data.iloc[:, 2]
    hotels_data = hotels_data.drop(columns=["NOM", "PAYS", "ETOILE"], index=1)  # pop non numeric data
    columns = hotels_data.columns
    data_array = hotels_data.to_numpy()
    scaler = StandardScaler()
    scaler.fit(data_array)  # Compute the mean and std to be used for later scaling. Necessary before transform
    fitted_data_array = scaler.transform(data_array)  # fit the data (mean = 0) and scale the data (std = 1)

    silhouetteScore = np.zeros(9)
    j = 0
    for i in range(2, 10):
        kmeans1 = KMeans(n_clusters=i, init='k-means++', n_init=80).fit(fitted_data_array)
        kmeans2 = KMeans(n_clusters=i, init='k-means++', n_init=80).fit(fitted_data_array)
        print("Adjusted Rand Score : ", adjusted_rand_score(kmeans1.labels_, kmeans2.labels_))
        silhouetteScore[j] = silhouette_score(fitted_data_array, kmeans1.labels_)
        print(silhouetteScore[j])
        j += 1

    values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.bar(values, silhouetteScore, width=0.02, color='green', label="Silhouette Scores")
    plt.show()
    kmeans1 = KMeans(n_clusters=5, init='k-means++', n_init=80).fit(fitted_data_array)
    print("KMeans  1 : ", kmeans1.labels_, kmeans1.inertia_)
    '''print("KMeans  1 : ", kmeans1.labels_, kmeans1.inertia_,
          "\nKMeans 2 : ", kmeans2.labels_, kmeans2.inertia_,
          "\nAdjusted Rand Score : ", adjusted_rand_score(kmeans1.labels_, kmeans2.labels_))'''
    for i in np.argsort(kmeans1.labels_):
        print("Nom : ", names[i], "; Label kmeans :", kmeans1.labels_[i], "; Stars : ", stars[i], "; Diff : ", kmeans1.labels_[i] + 1 - stars[i])


if __name__ == '__main__':
    hotels_data = pd.read_csv('hotels.csv')
    kmeans(hotels_data)
