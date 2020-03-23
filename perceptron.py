import numpy as np


class Perceptron:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import graph

    # irisデータセット読み込み
    df = pd.read_csv('https://archive.ics.uci.edu/ml/' 'machine-learning-databases/iris/iris.data', header=None)

    y = df.iloc[:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[:100, [0, 2]].values

    ppn = Perceptron(eta=0.1, n_iter=10)
    ppn.fit(X, y)

    # グラフ描画
    graph.plot_decision_regions(X, y, classifier=ppn)

    #plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    #plt.xlabel('Epochs')
    #plt.ylabel('Number of update')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


