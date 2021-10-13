import numpy as np
from numpy.linalg import norm


def function1(x):
    q = np.array([[2, 2]])
    P = np.array([[1, -1], [-1, 5]])
    xt = np.transpose(x)
    return 0.5 * (np.dot(np.dot(xt, P), x)) - np.dot(q, x)


def gradient1(x):
    P = np.array([[1, -1], [-1, 5]])
    q = np.array([2, 2])
    return np.dot(P, x) - q


def function2(x):
    x1, x2 = x
    return 10 * ((x1 * x1 - x2) * (x1 * x1 - x2)) + ((1 - x1) * (1 - x1))


def gradient2(x):
    x1, x2 = x
    dx1 = 40 * x1 * (x1 * x1 - x2) - 2 * (1 - x1)
    dx2 = - 20 * (x1 * x1 - x2)
    return np.array([dx1, dx2])


def dkOptimalGradient2(x):
    grad = gradient2(x)
    # print("Grad : ", grad)
    norm_ = norm(grad, 1)
    grad_for_max = abs(grad) / norm_
    # print("Normalisé : ", grad)
    max = grad_for_max.argmax()
    return grad[max]


def descenteGradient(x):
    fixed_step = 0.1
    stop = 0.01
    current_x = x
    nb_iterations = 0
    while norm(gradient1(current_x), 2) > stop:
        dk = -(gradient1(current_x))
        # step = optimumStep(current_x)
        wolf_step = wolfRules(current_x)
        current_x = current_x + wolf_step * dk
        # print(nb_iterations, " ; xk : ", current_x)
        nb_iterations += 1
    return current_x, function1(x), nb_iterations


def plusGrandeDescente(x):
    fixed_step = 0.1
    stop = 0.01
    current_x = x
    nb_iterations = 0
    while norm(gradient1(current_x), 1) > stop:
        dk = -dkOptimalGradient2(current_x)
        # step = optimumStep(current_x)
        # wolf_step = wolfRules(current_x)
        current_x = current_x + fixed_step * dk
        print(nb_iterations, " ; xk : ", current_x)
        nb_iterations += 1
    return current_x, function1(x), nb_iterations


def optimumStep(x):
    grad_x = gradient1(x)
    P = np.array([[1, -1], [-1, 5]])
    norm_grad_x = norm(grad_x, 2)
    result = (norm_grad_x * norm_grad_x) / (np.dot(np.dot(np.transpose(grad_x), np.transpose(P)), grad_x))
    # print("optimum step : ", result)
    return result


def wolfRules(x):
    # 0 < α1 < 0.5, 0.5 < β ≤ 0.9
    tk = 1
    alpha_1 = 0.5
    beta = 0.8
    dk = -gradient1(x)
    phi_tk_0 = phi(x, 0, dk)
    phi_prime_tk_0 = phi_prime(x, 0, dk)
    print("value : ", (phi_tk_0 + alpha_1 * phi_prime_tk_0 * tk))
    while phi(x, tk, dk) > (phi_tk_0 + alpha_1 * phi_prime_tk_0 * tk):
        tk = beta * tk
        print(tk)
    return tk


def phi(x, tk, dk):
    return function1(x + tk * dk)


def phi_prime(x, tk, dk):
    return np.dot(np.transpose(gradient1(x + tk * dk)), dk)


if __name__ == '__main__':
    x = np.array([2, 1])
    # print(function1(np.transpose(x)))
    # print(function2(np.transpose(x)))
    # print("gradient:", descenteGradient(np.transpose(x)))
    print("plus grande descente : ", plusGrandeDescente(np.transpose(x)))
