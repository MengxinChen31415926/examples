import numpy as np
from sklearn.decomposition import PCA

X = np.loadtxt('163_2.txt',delimiter=",", dtype=np.float32)
X=np.delete(X,-1,axis=1)

pca = PCA(n_components=32) # 降到32维
pca.fit(X)  # 训练
newX = pca.fit_transform(X)  # 降维后的数据
PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)  # 输出贡献率
print(newX)                  #输出降维后的数据

ft=open("pca2.txt","a+")
np.savetxt("pca2.txt", newX,fmt='%f',delimiter=',')
ft.close()