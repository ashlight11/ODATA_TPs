# This is a sample Python script.
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import linalg as lg
import sklearn
from sklearn.decomposition import PCA
from pandas import plotting as pg
from sklearn.preprocessing import StandardScaler

import csv

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


'''def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.'''


def generate_data():
    mean = [10, 20]
    cov = [[25, -12], [-12, 9]]
    matrix = np.random.multivariate_normal(mean, cov, 100)
    # plt.plot(matrix[0], matrix[1], 'x')
    # print(np.mean(matrix[1]))
    # plt.axis('equal')
    # plt.show()

    matrix = matrix - np.mean(matrix, axis=0)

    fig, (ax1, ax2) = plt.subplots(2)
    # plt.scatter(matrix[:, 0], matrix[:, 1])

    v = np.dot(matrix.T, matrix) / 100

    # cov = np.dot(matrix.T, )
    print(v)

    eig, eigenvectors = lg.eig(v)
    print(eig)
    print(eigenvectors)
    inertia = 100 * eig[0] / (eig[0] + eig[1])
    print(inertia)
    ax1.scatter(matrix[:, 0], matrix[:, 1])
    ax1.arrow(0, 0, eigenvectors[0, 0], eigenvectors[1, 0], head_width=0.2, head_length=0.3)
    ax1.arrow(0, 0, eigenvectors[0, 1], eigenvectors[1, 1], head_width=0.2, head_length=0.3)
    # plt.show()
    pca = PCA(n_components=1)
    pca.fit(matrix)
    print("PCA : ", pca.components_, pca.explained_variance_, pca.explained_variance_ratio_)

    reduction = pca.transform(matrix)

    ax2.scatter(reduction, np.zeros(100))
    # print("reduction", reduction)
    plt.show()


def activites():
    data = pd.read_csv('activites.txt', sep='\t')
    # print(data, data.shape)
    # data.info()

    # print(data)

    # groupe de personnes
    pop = data.iloc[:, 0]
    # print(pop)

    data = data.drop(columns=["POP", "SEX", "ACT", "CIV", "PAY"], index=1)
    # print(data)

    columns_for_acp = data.columns
    # print(columns_for_acp)

    corr = data.corr()
    # print(corr)
    pg.scatter_matrix(data, alpha=0.9)
    plt.show()

    matrix_x = data.to_numpy()
    scaler = StandardScaler()
    scaler.fit(matrix_x) # Compute the mean and std to be used for later scaling. Necessary before transform
    matrix_z = scaler.transform(matrix_x)  # fit the data (mean = 0) and scale the data (std = 1)
    print("Mean : ", scaler.mean_, "\nStd : ", scaler.var_)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    activites()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
