import pandas as pd
import numpy as np

class StandardPerceptron:

    def __init__(self, trainD, testD, r):
        w = self.perceptron(trainD.as_matrix(), 10, r)
        print(self.predict(testD.as_matrix(), w))

    def perceptron(self, D, T, r):

        w = np.zeros(4)
        for epoch in range(T):
            np.random.shuffle(D)

            for i in range(D.shape[0]):

                y = D[:, 4]
                x = D[:, :4]

                error = y[i]*(w.dot(x[i]))
                if error <= 0:
                    w = w + (r*(y[i]*x[i]))
        return w

    def predict(self, D, w):

        errorSum = 0
        for i in range(D.shape[0]):
            x = D[:, :4]
            y = D[:, 4]

            error = y[i]*(w.T.dot(x[i]))

            if error <= 0:
                errorSum += 1

        return errorSum/D.shape[0]

class VotedPerceptron:

    def __init__(self, trainD, testD, r):
        w, c = self.perceptron(trainD.as_matrix(), 10, r)
        print('w len = {0}'.format(len(w)))
        print(self.prediction(testD.as_matrix(), w, c))

    def perceptron(self, D, T, r):

        w = [np.zeros(4)]
        c = [1]
        m = 0
        for epoch in range(T):
            np.random.shuffle(D)

            for i in range(D.shape[0]):

                y = D[:, 4]
                x = D[:, :4]

                error = y[i]*(w[m].dot(x[i]))
                if error <= 0:
                    # w[m] = w + (r*(y[i]*x[i]))
                    w.append(w[m] + (r*(y[i]*x[i])))
                    m += 1
                    c.append(1)
                else:
                    c[m] += 1

        return w, c

    def prediction(self, D, w, c):

        errorSum = 0
        for i in range(D.shape[0]):
            x = D[:, :4]
            y = D[:, 4]

            sgn = sum([ c[j] * w[j].dot(x[i]) for j in range(len(w))])

            if sgn/abs(sgn) != y[i]:
                errorSum += 1

        return errorSum/D.shape[0]

class AveragePerceptron:

    def __init__(self, trainD, testD, r):
        w = self.perceptron(trainD.as_matrix(), 10, r)
        print(self.prediction(testD.as_matrix(), w))

    def perceptron(self, D, T, r):

        w = np.zeros(4)
        a = np.zeros(4)
        for epoch in range(T):
            np.random.shuffle(D)

            for i in range(D.shape[0]):

                y = D[:, 4]
                x = D[:, :4]

                error = y[i]*(w.dot(x[i]))
                if error <= 0:
                    w = w + (r*(y[i]*x[i]))

                a = a + w
        return w

    def prediction(self, D, a):

        errorSum = 0
        for i in range(D.shape[0]):
            x = D[:, :4]
            y = D[:, 4]

            error = (a.dot(x[i]))

            if error/abs(error) != y[i]:
                errorSum += 1

        return errorSum/D.shape[0]


if __name__ == '__main__':


    dt = pd.read_csv('../dataset-hw2/classification/train.csv')
    dt.columns = ['x1', 'x2', 'x3', 'x4', 'y']
    dt.loc[dt['y'] == 0] = -1

    testD = pd.read_csv('../dataset-hw2/classification/test.csv')
    testD.columns = ['x1', 'x2', 'x3', 'x4', 'y']
    testD.loc[dt['y'] == 0] = -1

    r = 1/10000
    StandardPerceptron(dt, testD, r)
    VotedPerceptron(dt, testD, r)
    AveragePerceptron(dt, testD, r)

