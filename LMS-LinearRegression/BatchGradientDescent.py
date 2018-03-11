import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import pandas as pd

J = []
def BatchGradientDescennt(x, y, r, e):

    w = np.transpose(np.zeros(x.shape[1]))
    m = x.shape[0]

    b = 0

    counter = 0
    while True:

        grad_w = sum([ (y[i] - w.dot(x[i])) * x[i] for i in range(m)])

        w_t1 = w + (r * (grad_w))

        if np.linalg.norm((w_t1) - (w)) <= e:
            print('Converged after {0} runs'.format(counter))
            break

        w = w_t1
        J.append ((1/2) * sum([ (y[i] - w.dot(x[i]) - b)**2 for i in range(m) ]))

        counter += 1

    print('w = {0}'.format(w))
    print('b = {0}'.format(b))
    return w, b


if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                           random_state=0, noise=35)

    x = np.array([[1, -1, 2],
                  [1, 1, 3],
                  [-1, 1, 0],
                  [1, 2, -4],
                  [3, -1, -1]])

    y = np.array([[1], [4], [-1], [-2], [0]])

    dt = pd.read_csv('../dataset-hw2/regression/train.csv')
    dt.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'y']
    dt['b'] = 1
    xpd = dt.loc[:, ['b','x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7']]
    ypd = dt.loc[:,['y']]

    x = xpd.as_matrix()
    y = ypd.values
    r = 0.0625/4 # learning rate
    ep = 1/1000000 # convergence criteria

    b = 0

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    w, b = BatchGradientDescennt(x, y, r, ep)#, x, y, ep, max_iter=1000)
    print (('w = %s b = %s') %(w, b))

    pylab.plot(J)
    pylab.show()

    print ("Done!")


# x = np.array([[1, -1, 2],
    #               [1, 1, 3],
    #               [-1, 1, 0],
    #               [1, 2, -4],
    #               [3, -1, -1]])
    #
    # y = np.array([[1], [4], [-1], [-2], [0]])


    # w = np.transpose(np.zeros(3))
    # b = 0
    #
    # r = 0.1
    # e = 0.1
    # w = np.transpose(np.array([-1, 1, -1]))
    # b = -1
    #
    # w = np.transpose(np.array([1/2, -1/2, 1/2]))
    # b = 1

    # m = x.shape[0]
    # m1 = x.shape[1]
    #
    # wm = w.shape[0]
    # wT = np.transpose(w)
    #
    # newW = np.ones(wm)
    # for j in range(wm):
    #     res = []
    #     for i in range(m):
    #         # print('wT.dot(x[i] = {0}'.format(wT.dot(x[i])))
    #         # print('({0} - {1})*{2}'.format(y[i], wT.dot(x[i]), x[i][j]))
    #         res.append((y[i] - wT.dot(x[i]) - b) * x[i][j])
    #     newW[j] = (-1 * sum(res))
    #
    # update = newW - (0.01 * w)
    # print(newW)
    #
    # print('\n'*3)

    # BatchGradientDescennt(x, y, b, r, e)