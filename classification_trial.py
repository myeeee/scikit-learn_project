from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import graph

# iris dataset 読み込み
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# train, testにdataを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# データの標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# パーセプトロン
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_test_std)

# ロジスティック回帰
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

#print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

# trainとtestを結合する
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

graph.plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()