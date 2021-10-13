import time
from datetime import timedelta

import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree

if __name__ == '__main__':
    '''data_d = pd.DataFrame(data=[(18, 2),
                                (3, 19),
                                (8, 19),
                                (4, 1),
                                (2, 5),
                                (8, 5),
                                (14, 9),
                                (11, 15),
                                ], columns=["Valeur 1", "Valeur 2"])

    # print(data_d)
    data_test = data_d.to_numpy()
    query_q = np.array([6, 10])
    query_q = query_q.reshape(1, -1)
    nodes_values = [(1, 7),
                    (0, 2),
                    (1, 17),
                    (1, 3),
                    (1, 3),
                    (0, 13),
                    (1, 12),
                    (0, 5.5)]
    data_kd = np.array(nodes_values)
    kdt = KDTree(data_test, leaf_size=30, metric='euclidean')
    dist, ind = kdt.query(query_q, k=2, return_distance=True)
    print("distances : ", dist, "\nindices : ", ind)'''

    dataset_vectors_200d = np.load('glove_200d/dataset_glove_200d.npy')
    # dataset_words_200d = np.load('glove_200d/dataset_words_glove_200d.npy')
    probes_vectors = np.load('glove_200d/probes_glove_200d.npy')
    probes_words = np.load('glove_200d/probes_words_glove_200d.npy')

    # print("Dataset vector representation for words shape : ", dataset_vectors_200d.shape,

    bf = NearestNeighbors(algorithm="brute")
    '''bf.fit(dataset_vectors_200d)
    query = probes_vectors[0:2]
    result = bf.kneighbors(query) 
    nearest_first = result[0][1]
    # print(dataset_words_200d[nearest_first])
    nearest_second = result[1][1]
    # print(dataset_words_200d[nearest_second])'''

    kt = NearestNeighbors(algorithm="kd_tree")
    bt = NearestNeighbors(algorithm="ball_tree")
    algorithms = [('brute', bf), ('kd-tree', kt), ('ball-tree', bt)]

    start_build = time.monotonic()
    bf.fit(dataset_vectors_200d)
    end_build = time.monotonic()
    print("build time : ", timedelta(seconds=end_build - start_build))
    start_search = time.monotonic()
    result = bf.kneighbors(probes_vectors[:20])
    # result = bf.kneighbors(probes_vectors) Pour les 100 requêtes
    end_search = time.monotonic()
    print("search time : ", timedelta(seconds=end_search - start_search))
'''for name, alg in algorithms:
        print(name)
        start_build = time.monotonic()
        alg.fit(dataset_vectors_200d)
        end_build = time.monotonic()
        print("build time : ", timedelta(seconds=end_build - start_build))
        start_search = time.monotonic()
        result = alg.kneighbors(probes_vectors[0].reshape(1, -1))
        end_search = time.monotonic()
        print("search time : ", timedelta(seconds=end_search - start_search))
        print(result) '''

# Pour mesurer la durée d'éxécution en fonction de k, faire boucle for k in [2, 5, 10, 20]
