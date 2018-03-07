import numpy as np
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats

def BatchGradientDescennt(x, y, r, e):

    w = np.transpose(np.zeros(x.shape[1]))
    m = x.shape[0]

    b = 0

    J = (1/2) * sum([ (y[i] - w.dot(x[i]) - b)**2 for i in range(m) ])
    counter = 0
    while True:

        grad_w = sum([ (y[i] - w.dot(x[i])) * x[i] for i in range(m)])
        # grad_b = sum([ (y[i] - w.dot(x[i]) - b) for i in range(m)])
        w_t1 = w + (r * (grad_w))
        # b_t1 = b - (r * (grad_b))

        # print(np.linalg.norm(w_t1 - w))
        # print(w_t1 - w)
        # print(e)
        # print(np.linalg.norm(w - w_t1) <= e)

        # error = (1/2) * sum([ (y[i] - w_t1.dot(x[i]) - b_t1)**2 for i in range(m) ])

        if np.linalg.norm(w_t1 - w) <= e:
        # if abs(error - J) <= e:
            print('Converged after {0} runs'.format(counter))
            break

        w = w_t1
        # b = b_t1


        counter += 1
        print(counter)

    print('w = {0}'.format(w))
    print('b = {0}'.format(b))
    return w, b

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
                           random_state=0, noise=35)
    # print ('x.shape = %s y.shape = %s' %(x.shape, y.shape))
    #

    x = np.array([[1, -1, 2],
                  [1, 1, 3],
                  [-1, 1, 0],
                  [1, 2, -4],
                  [3, -1, -1]])

    y = np.array([[1], [4], [-1], [-2], [0]])

    r = 0.0625 # learning rate
    ep = 1/1000000 # convergence criteria

    b = 0

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    theta0, theta1 = BatchGradientDescennt(x, y, r, ep)#, x, y, ep, max_iter=1000)
    print (('theta0 = %s theta1 = %s') %(theta0, theta1))

    try:
        # check with scipy linear regression
        slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
        print (('intercept = %s slope = %s') %(intercept, slope))

        y_predict = 0
        # plot
        for i in range(x.shape[0]):
            y_predict = theta0 + theta1*x

        pylab.plot(x,y,'o')
        pylab.plot(x,y_predict,'k-')
        pylab.show()
    except:
        pass
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