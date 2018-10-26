import numpy as np

def data_load(path):
    f = open(path)
    try:
        lines = f.readlines()
    finally:
        f.close()

    example_num = len(lines)
    feature_dimension = len(lines[0].strip().split())

    features = np.zeros((example_num, feature_dimension))
    features[:, 0] = 1
    labels = np.zeros((example_num, 1))

    for index, line in enumerate(lines):
        items = line.strip().split(' ')
        features[index, 1:] = [float(str_num) for str_num in items[0: -1]]
        labels[index] = float(items[-1])

    return features, labels

def gradient_descent(X, y, w):
    tmp = -y * (np.dot(X, w))
    weight_matrix = np.exp(tmp)/(1 + np.exp(tmp) * 1.0)
    gradient = 1/(len(X) * 1.0) * (sum(weight_matrix * (-y) * X).reshape(len(w), 1))
    return gradient

#gradient descent
def stochastic_gradient_descent(X, y, w):
    tmp = -y * (np.dot(X, w))
    weight = np.exp(tmp)/((1 + np.exp(tmp)) * 1.0)
    gradient = weight * (-y) * X
    return gradient.reshape(len(gradient), 1)

class LinearRegression:
    def __init__(self):
        pass

    def fit(self, X, y, Eta = 0.001, max_iter = 2000, sgd = True):
        self.__w = np.zeros((len(X[0]), 1))
        if not sgd:
            for i in range(max_iter):
                self.__w = self.__w - Eta * gradient_descent(X, y, self.__w)
        else:
            index = 0
            for i in range(max_iter):
                if (index >= len(X)):
                    index = 0
                self.__w = self.__w - Eta * stochastic_gradient_descent(np.array(X[index]), y[index], self.__w)
                index += 1

    def predict(self, X):
        binary_result = np.dot(X, self.__w) >= 0
        return  np.array([(1 if _ > 0 else -1) for _ in binary_result]).reshape(len(X), 1)

    def get_w(self):
        return self.__w

    def score(self, X, y):
        predict_y = self.predict(X)
        return sum(predict_y != y)/(len(y) * 1.0)

if __name__ == '__main__':
    (X, Y) = data_load('hw3_train.dat')
    lr = LinearRegression()
    lr.fit(X, Y, max_iter=2000)

    test_X, test_Y = data_load('hw3_test.dat')
    print('Eout:', lr.score(test_X, test_Y))




