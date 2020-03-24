import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


# Wineデータセットの読み込み
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# train用とtest用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

# データ標準化
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# 共分散行列
cov_mat = np.cov(X_train_std.T)
# 固有値, 固有ベクトル
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# 描画用
tot = sum(eigen_vals)
# 各固有値の分散説明率
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# 分散説明率の累積和
cum_var_exp = np.cumsum(var_exp)

# 固有値大きい順にソート
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
# 射影行列
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis]))
# 変換
X_train_pca = X_train_std.dot(w)

colors = ['r','b','g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train==l, 0], X_train_pca[y_train==l, 1], c=c, label=l, marker=m)


#plt.bar(range(1, 14), var_exp, alpha= 0.5, align='center', label='individual explained variance')
#plt.step(range(1, 14), cum_var_exp, where='mid', label='cumulative explained variance')
plt.legend(loc='best')
plt.tight_layout()
plt.show()