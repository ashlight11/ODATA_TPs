import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_distances(data, query, dist):
    computed_distance = np.zeros(len(data), dtype="float64")
    if dist == "L2":
        computed_distance = np.sqrt(np.sum(np.square(data - query), axis=1))
    elif dist == "L1":
        computed_distance = np.sum(np.abs(data - query), axis=1)
    elif dist == "L_infinite":
        computed_distance = np.max(np.abs(data - query), axis=1)
    return computed_distance


def onenn_search(data, query, dist):
    all_distances = compute_distances(data, query, dist)
    return np.argmin(all_distances), all_distances[np.argmin(all_distances)]


def knn_search(data, query, k, dist):
    nearest_neighbours = np.sort(compute_distances(data, query, dist))[0:k]
    return nearest_neighbours


def prof_knn_search(data, query, k, dist):
    distances = compute_distances(data, query, dist)
    nearest = []
    for i, d in enumerate(distances):
        if i < k:
            nearest.append((d, i))
            nearest.sort()
        else:
            max_dist = nearest[-1][0]
            if d < max_dist:
                nearest.pop(-1)
                nearest.append((d, i))
                nearest.sort()
    return nearest




if __name__ == '__main__':
    data_d = pd.DataFrame(data=[(3.0, 3.5),
                                (2.0, 1.0),
                                (3.0, 5.5),
                                (6.0, 2.0),
                                (6.0, 5.5),
                                (1.0, 5.0),
                                (6.0, 4.0),
                                (2.5, 3.0),
                                (4.0, 2.5),
                                (5.0, 5.0)
                                ], columns=["Valeur 1", "Valeur 2"])

    # print(data_d)
    data_test = data_d.to_numpy()
    request_q = (4.0, 4.0)
    plt.scatter(data_d.iloc[:, 0], data_d.iloc[:, 1], color="blue")
    plt.scatter(request_q[0], request_q[1], color="red")
    # plt.show()

    print("norme L1 : ", compute_distances(data_d, request_q, "L1"),
          "\nnorme L2: ", compute_distances(data_d, request_q, "L2"),
          "\nnorme L_infini: ", compute_distances(data_d, request_q, "L_infinite"))

    print("KNN search L2 : ", onenn_search(data_test, request_q, "L2"),
          "\nKNN search L2 : ", prof_knn_search(data_test, request_q, 3, "L2"))

    dataset_ratings_1 = np.load("npy_tp1/dataset_ratings_1.npy")
    dataset_titles_1 = np.load("npy_tp1/dataset_titles_1.npy")
    probes_ratings_1 = np.load("npy_tp1/probes_ratings_1.npy")
    probes_titles_1 = np.load("npy_tp1/probes_titles_1.npy")

    # print("dataset : ", dataset_ratings_1.shape, "  probes : ", probes_ratings_1.shape)
    nearest_neighbours = prof_knn_search(dataset_ratings_1[:500], probes_ratings_1[0], 10, "L2")
    print("Film initial : ", probes_titles_1[0])

