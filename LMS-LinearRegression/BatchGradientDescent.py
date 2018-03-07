import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats


def Error(x, y, t0, t1, m):
    sm = 0.0
    for i in range(m):
        sm += (1.0 / m) * (y[i] + x[i] * t1 + t0) ** 2

    return sm


def GradientW(x, y, w, b, m):
    s = sum([(y[i] - w * x[i] + b) * x[i] for i in range(m)])
    return -1 * s
    # s = 0.0
    # for i in range(m):
    #     s += (y[i] - w*x[i] + b)#*x[i]
    # return -1 * s#sum([  for i in range(m)])


def GradientB(x, y, w, b, m):
    s = sum([(y[i] - w * x[i] + b) for i in range(m)])
    return -1 * s


def Batch_Gradient_Descent(x, y, r):
    m = x.shape[0]
    w = np.random.random(x.shape[1])
    b = np.random.random(x.shape[1])

    # J = Error(x, y, t0, t1, m)
    J = sum([(y[i] - b + w * x[i]) ** 2 for i in range(m)])

    it = 0
    while True:

        # grad_of_0 = -1 * (1.0/m)*sum([ (y[i] -  (t1*x[i]) + t0) for i in range(m) ])
        # grad_of_1 = #Gradient(x, y, t0, t1, m)
        # grad_of_b = -1 * (sum([(y[i] - b + w*x[i]) for i in range(m)]))
        # grad_of_w = -1 * (sum([(y[i] - b + w*x[i])*x[i] for i in range(m)]))

        d = np.dot(w, x)

        w = w - (r * grad_of_w)
        b = b - (r * grad_of_b)

        # error = Error(x, y, t0, t1, m)
        error = sum([(y[i] - b + w * x[i]) ** 2 for i in range(m)])

        if abs(J - error) <= r:
            print('Converged, iterations: ', it, '!!!')
            print("Converged")
            break
        it += 1
        J = error
    return b, w


def BatchGradientDescennt(x, y, b, r, e):

    w = np.transpose(np.zeros(x.shape[1]))
    m = x.shape[0]

    J = (1/2) * sum([ (y[i] - w.dot(x[i]) - b)**2 for i in range(m) ])
    counter = 0
    while True:
        grad_w = 0.0
        grad_b = 0.0
        for i in range(m):
            # my_lst_str = ','.join(map(str, x[i]))
            # print(' &+ ({0} - {1} - {2}) * [{3}]\\\\'.format(y[i][0], wT.dot(x[i]), b, my_lst_str))
            grad_w += (y[i] - w.dot(x[i]) - b) * x[i]
        # print('&= {0}'.format(-1 * grad_w))
        # print()

        for i in range(m):
            # print(' &+ ({0} - {1} - {2})\\\\'.format(y[i][0], wT.dot(x[i]), b))
            grad_b += (y[i] - w.dot(x[i]) - b)

        # print('&= {0}'.format(-1 * grad_b))
        w = w - r*(-1 * grad_w)
        b = (b - r*(-1 * grad_b))#[0]

        error = (1/2) * sum([ (y[i] - w.dot(x[i]) - b)**2 for i in range(m) ])
        if np.isnan(error):
            print()

        if counter == 912:
            print()

        if abs(J - error) <= e:
            print('Converged after {0} runs'.format(counter))
            break

        J = error
        counter += 1
        print(counter)

    print('w = {0}'.format(w))
    print('b = {0}'.format(b))
    return w, b

if __name__ == '__main__':

    # x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
    #                        random_state=0, noise=35)
    # print ('x.shape = %s y.shape = %s' %(x.shape, y.shape))
    #

    x = np.array([[1, -1, 2],
                  [1, 1, 3],
                  [-1, 1, 0],
                  [1, 2, -4],
                  [3, -1, -1]])

    y = np.array([[1], [4], [-1], [-2], [0]])

    r = 0.01 # learning rate
    ep = 1/1000000 # convergence criteria

    b = 0

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    theta0, theta1 = BatchGradientDescennt(x, y, b, r, ep)#, x, y, ep, max_iter=1000)
    print (('theta0 = %s theta1 = %s') %(theta0, theta1))

    # check with scipy linear regression
    # slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
    # print (('intercept = %s slope = %s') %(intercept, slope))

    # y_predict = 0
    # # plot
    # for i in range(x.shape[0]):
    #     y_predict = theta0 + theta1*x
    #
    # pylab.plot(x,y,'o')
    # pylab.plot(x,y_predict,'k-')
    # pylab.show()
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