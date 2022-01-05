import math

import numpy as np
from numpy.linalg import inv


def func(x):
    f = np.zeros([n])
    for i in range(n):
        f[i] = x[i] * math.log(x[i])
    return np.sum(f)


def gradf(x):
    df = np.zeros([n])
    for i in range(n):
        df[i] = math.log(x[i]) + 1
    return df.transpose()


def Hf(x):
    H = np.zeros([n, n])
    for i in range(n):
        # H[i][i] = 1/x[i]
        H[i, i] = 1 / x[i]
    return H


def phi(x, tk, dk):
    return func(x + tk * dk)


def phi_prime(x, tk, dk):
    return np.dot(np.transpose(gradf(x + tk * dk)), dk)


def wolfRules(x):
    # 0 < α1 < 0.5, 0.5 < β ≤ 0.9
    tk = 1
    alpha_1 = 0.2
    beta = 0.6
    dk = -gradf(x)
    phi_tk = phi(x, 0, dk)
    phi_prime_tk_0 = phi_prime(x, 0, dk)
    seuil = phi_prime_tk_0 + alpha_1 * phi_prime_tk_0 * tk
    # print("value : ", (phi_tk_0 + alpha_1 * phi_prime_tk_0 * tk))
    while phi_tk > seuil:
        tk = beta * tk
        x = x + tk * dk
        phi_tk = phi(x, tk, dk)
        phi_prime_tk = phi_prime(x, 0, dk)
        seuil = phi_prime_tk + alpha_1 * phi_prime_tk * tk
        # print(tk)
    return tk


if __name__ == '__main__':
    b = 0
    somme_proba = 0
    n = 64
    eps = 10e-12
    x0 = np.ones([n])
    ro = np.array([])
    for i in range(n):
        # compute the costs
        tmp = 1 - math.exp(-(i + 1) * 0.02)  # note that i takes values from 0 to (n-1)
        ro = np.append(ro, tmp)  # array with all the costs
        # compute the probabilities
        x0[i] = math.exp(-(i + 1) * 0.08)  # note that i takes values from 0 to (n-1)

    # print(ro)
    x0 = x0 / np.sum(x0)  # fit the probabilities so that sum x0 = 1
    # print(x0)
    print(np.sum(x0))
    # print(np.dot(x0,ro))
    b = np.dot(x0, ro)
    print("b = ", b)

    Hf0 = Hf(x0)
    A = np.ones([2, n])
    A[1, :] = ro
    matrix = np.zeros([n + 2, n + 2])
    matrix[:n, :n] = Hf0
    matrix[:n, n:] = np.transpose(A)
    matrix[n:, :n] = A


    Hfk = Hf(x0)
    Hfinv = np.linalg.inv(Hfk)
    # print(Hf0inv)
    # dx = -Hf0inv*(gradf(x0).transpose())
    gradfk = gradf(x0)
    dx = -np.dot(Hfinv, gradfk)
    # print(dx)
    criterion = np.dot(np.dot(dx, Hfk), dx) / 2
    # print("criterion: ",criterion)
    xk = x0
    i = 0
    t = 1

    while (criterion > eps):
        # for i in range(1):
        M = matrix
        invM = np.linalg.inv(M)
        arr_tmp = np.array([0, 0])
        TT = np.append(gradfk, arr_tmp)
        C = -np.dot(invM, TT)
        # print(C)
        dxk = C[0:n]
        # print(dxk)
        vk = C[n:n + 1]
        # print(vk)
        ## Use Wolfe condition to search for the step (le pas)
        t = 1
        dphi0 = np.dot(gradfk.transpose(), dxk)
        pas = wolfRules(xk)
        # print("pas : ", pas)
        xopta = xk
        # print(np.dot(A,dxk))
        # print("b:",np.dot(A,xk))
        # print("sum(xk)",np.sum(xk))
        ##update
        xk = xk + pas * dxk
        gradfk = gradf(xk)
        # print(func(xk))
        Hfk = Hf(xk)
        criterion = np.dot(np.dot(dxk, Hfk), dxk) / 2
        i = i + 1
        Niter = i
        # print("Niter: ",Niter)
        # xopta = xk

    print("Niter: ", Niter)
    # print("xopt: ",xopta)
    fopt = func(xopta)
    print('fmin = ', fopt)
    print('Entrpoy = fmax = ', -fopt)
    print("sum(xopt) = ", np.sum(xopta))
    print("<xopta,ro> = ", np.dot(xopta, ro))
    print("b = ", b)
