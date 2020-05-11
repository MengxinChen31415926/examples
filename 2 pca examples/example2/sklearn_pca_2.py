import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

X = np.loadtxt('163_2.txt',delimiter=",", dtype=np.float32)
X=np.delete(X,-1,axis=1)

# 绘图，展示主成分的累计贡献率随降维维度增长的趋势图
def plot_durations(m,n):
    plt.figure(1)
    plt.clf()
    # 行，列，索引
    plt.subplot(1,1,1)
    plt.plot(m+1,n)
    plt.pause(0.001)  # pause a bit so that plots are updated

m=[]
n=[]

print(X)

for i in range(76):
    total = 0

    pca = PCA(n_components=i)
    pca.fit(X)  # 训练
    newX = pca.fit_transform(X)  # 降维后的数据
    # PCA(copy=True, n_components=2, whiten=False)
    # print(pca.explained_variance_ratio_)  # 输出贡献率
    # print(newX)                  #输出降维后的数据

    for ele in range(0, len(pca.explained_variance_ratio_)):
        total = total + pca.explained_variance_ratio_[ele]

    print(i+1,"\t",total)
    # if (total>=0.8):
    #     break;
    n.append(np.array(total))
    m.append(np.array(i+1))
    plot_durations(np.array(m), np.array(n))

input("请输入任意键以继续......")