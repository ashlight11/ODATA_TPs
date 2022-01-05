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
    # pg.scatter_matrix(data, alpha=0.9)
    # plt.show()

    matrix_x = data.to_numpy()
    scaler = StandardScaler()
    scaler.fit(matrix_x)  # Compute the mean and std to be used for later scaling. Necessary before transform
    matrix_z = scaler.transform(matrix_x)  # fit the data (mean = 0) and scale the data (std = 1)
    print("Mean : ", np.mean(matrix_z), "\nStd : ", np.var(matrix_z))

    pca_ten_values = PCA()
    pca_ten_values.fit(matrix_z)
    # print("PCA components (principal axes: ", pca.components_, "\nPCA explained variance (eigenvalues): ",
    # pca.explained_variance_, "\nPCA explained variance ratio (Percentage of variance explained by each of the
    # selected components): ", pca.explained_variance_ratio_)
    cumsum = np.cumsum(pca_ten_values.explained_variance_)
    eig_val_desc = -np.sort(-pca_ten_values.explained_variance_)
    explained_variance_ratio_desc = -np.sort(-pca_ten_values.explained_variance_ratio_)
    '''print("Eigenvalues descending order : ", eig_val_desc,
          "\nExplained variance ratio descending order : ", explained_variance_ratio_desc * 100,
          "\nCumulated sum for inertia : ", cumsum)'''

    x_eigenvalues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # plt.plot(cumsum, "r-", label='Cumulated sum')
    # plt.bar(x_eigenvalues, eig_val_desc, width=0.02, color='green', label="Descending eigenvalues")
    # plt.show()

    pca_4_values = PCA(n_components=4)
    pca_4_values.fit(matrix_z)
    eigenvalues = pca_4_values.explained_variance_
    print("Global quality with 4 axes kept : ", np.sum(eigenvalues) / np.sum(eig_val_desc))
    matrix_z = pca_4_values.transform(matrix_z)
    pop_np = pop.to_numpy()
    # print(matrix_z)
    '''fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(matrix_z[:, 0], matrix_z[:, 1])
    ax2.scatter(matrix_z[:, 2], matrix_z[:, 3])
    
    for i in range(0, len(pop_np) - 1):
        ax1.text(matrix_z[i, 0], matrix_z[i, 1], pop_np[i])
        ax2.text(matrix_z[i, 2], matrix_z[i, 3], pop_np[i])
    ax1.title.set_text("Axes 1 and 2")
    ax2.title.set_text("Axes 3 and 4")'''

    for j in range(0, 4):
        contribution_to_every_axis = (1 / len(pop_np)) * matrix_z[:, j] * (1 / eigenvalues[j])
        ind_max = np.argmax(contribution_to_every_axis)
        print("Contribution to axis ", j + 1, " ", contribution_to_every_axis,
              "\nArgmax : ", ind_max, " Nom : ", pop[ind_max], "\nValeur : ", contribution_to_every_axis[ind_max])

    # plt.show()

    four_eigenvectors = pca_4_values.components_
    # print("Eigenvectors : ", four_eigenvectors)
    four_first_principal_factors = np.zeros(4)
    for k in range(0, 4):
        four_first_principal_factors[k] = matrix_z[0, k] / np.sqrt(eig_val_desc[k])
    print("Principal factors : ", four_first_principal_factors)

    # corrélation des variables avec les axes
    corvar = np.zeros((10, 4))

    for k in range(4):
        corvar[:, k] = pca_4_values.components_[k, :] * np.sqrt(eig_val_desc[k])
        print(pca_4_values.components_[k, :])

    # afficher la matrice des corrélations variables x facteurs
    print(corvar)

    fig, axes = plt.subplots(figsize=(8, 8))
    axes.set_xlim(-1, 1)
    axes.set_ylim(-1, 1)

    cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
    axes.add_artist(cercle)
    for j in range(10):
        plt.annotate(columns_for_acp[j], (corvar[j, 0], corvar[j, 1]))

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    activites()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
