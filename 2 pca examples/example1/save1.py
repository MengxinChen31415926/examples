import numpy as np
from sklearn.decomposition import PCA

X = np.loadtxt('163_1.txt')

pca = PCA(n_components=15)  # 降到15维
pca.fit(X)  # 训练
newX = pca.fit_transform(X)  # 降维后的数据
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出贡献率
print(newX)                  #输出降维后的数据

ft=open("pca1.txt","a+")
np.savetxt("pca1.txt", newX,fmt='%f',delimiter=',')
ft.close()