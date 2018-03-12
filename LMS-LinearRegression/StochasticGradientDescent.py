# import numpy as np
#
# r = 0.1
# x = np.array([[1, -1, 2],
#               [1, 1, 3],
#               [-1, 1, 0],
#               [1, 2, -4],
#               [3, -1, -1]])
#
# y = np.array([[1], [4], [-1], [-2], [0]])
#
# w = np.transpose(np.zeros(3))
# b = np.zeros(1)
#
# m = x.shape[0]
#
# for i in range(m):
#
#     temW = np.zeros(w.shape[0])
#     for j in range(w.shape[0]):
#         print('$w\\big[{6}\\big] = {0} + {1}*({2} - {3} - {4})*{5}$\\\\'.
#               format(format(w[j], '.4f'), r, y[i][0], format(w.dot(x[i]), '.4f'), format(b[0], '.4f'), x[i][j], j))
#         temW[j] = w[j] + r * (y[i] - w.dot(x[i]) - b) * x[i][j]
#
#     print('$\w = {0}$\\\\'.format(temW))
#
#     print('$b = {0} + {1}*({2} - {3} - {4})$\\\\'.format(format(b[0], '.4f'), r, format(y[i][0], '.4f'),
#                                                    format(w.dot(x[i]), '.4f'), format(b[0], '.4f')))
#     b = b + r * (y[i] - w.dot(x[i]) - b)
#     print('$b = {0}$\\\\'.format(format(b[0], '.4f')))
#     print()
#
#     w = temW

import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
import pandas as pd

J = []
def StochasticGradientDescent(x, y, r, e, iterations):

    w = np.transpose(np.zeros(x.shape[1]))
    m = x.shape[0]

    counter = 1

    # j = (1/2) * sum([ (y[i] - w.dot(x[i]))**2 for i in range(m) ])
    J.append((1/2) * sum([(y[i] - w.dot(x[i]))**2 for i in range(m)])[0])

    while True:

        # for i in range(m):
        i = random.randint(0, m-1)
        temW = np.zeros(w.shape[0])
        for j in range(w.shape[0]):
            temW[j] = w[j] + (r * (y[i] - w.dot(x[i])) * x[i][j])

        w = temW


        J.append ((1/2)*sum([y[i] - temW.dot(x[i])**2 for i in range(m)])[0])

        if abs(J[counter] - J[counter-1]) <= e:
            print('Converged after {0} runs'.format(counter))
            return w

        counter += 1


        if iterations <= counter:
            return None


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
    r = 1#/(64*2*2**2*2*2*2) # learning rate
    ep = 1/1000000 # convergence criteria

    b = 0

    w = []
    while r > ep:
        J = []
        w = StochasticGradientDescent(x, y, r, ep, 1000)

        if w is not None:
            break
        r /= 2

    # w = StochasticGradientDescent(x, y, r, ep, 1000)

    print (('w = {0}, b = {1}').format(w, b))
    print('r = {0}'.format(r))
    print('Final Cost Funtion = {0}'.format((1/2) * sum([ (y[i] - w.dot(x[i]))**2 for i in range(x.shape[0]) ])))

    print(J[1:])
    pylab.plot(J[1:])
    pylab.title('Stocastic Gradient Descent')
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