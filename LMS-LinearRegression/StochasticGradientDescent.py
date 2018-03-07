import numpy as np

r = 0.1
x = np.array([[1, -1, 2],
              [1, 1, 3],
              [-1, 1, 0],
              [1, 2, -4],
              [3, -1, -1]])

y = np.array([[1], [4], [-1], [-2], [0]])

w = np.transpose(np.zeros(3))
b = np.zeros(1)

m = x.shape[0]

for i in range(m):

    temW = np.zeros(w.shape[0])
    for j in range(w.shape[0]):
        print('$w\\big[{6}\\big] = {0} + {1}*({2} - {3} - {4})*{5}$\\\\'.
              format(format(w[j], '.4f'), r, y[i][0], format(w.dot(x[i]), '.4f'), format(b[0], '.4f'), x[i][j], j))
        temW[j] = w[j] + r * (y[i] - w.dot(x[i]) - b) * x[i][j]

    print('$\w = {0}$\\\\'.format(temW))

    print('$b = {0} + {1}*({2} - {3} - {4})$\\\\'.format(format(b[0], '.4f'), r, format(y[i][0], '.4f'),
                                                   format(w.dot(x[i]), '.4f'), format(b[0], '.4f')))
    b = b + r * (y[i] - w.dot(x[i]) - b)
    print('$b = {0}$\\\\'.format(format(b[0], '.4f')))
    print()

    w = temW
