from sklearn.linear_model import LogisticRegression
#调用机器学习库中的逻辑回归模型 若无skic-learn，命令行输入pip install sklearn 安装
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


#数据准备-鸢尾花数据集 可直接联网下载 不用本地导入
iris = datasets.load_iris()
list(iris.keys())
X1=iris["data"][:,2:]  #petal width花瓣宽度
y1 = (iris["target"] == 0).astype(int)#1 if Iris_Virginica, else 0是鸢尾花为1，否为0
data2=pd.DataFrame(X1, columns=["A","B"]) #训练数据的X（包含A/B两个特征）
data2["C"]=y1 #训练数据的Y

# 模型建立和预测
log_reg = LogisticRegression() #调用逻辑回归库
log_reg.fit(X1,y1) #训练模型
#print(a)
print(log_reg.intercept_,log_reg.coef_) #输出预测参数
#plt.scatter(X1[:,0],X1[:,1])
log_reg.predict([[1.5,1.7]])#单例预测，预测（A=1.5，B=1.7时的C（Y）为？
log_reg.score(X1,y1)#模型训练结果评估，输出准确率

# 将分类器绘制到图中
def plot_classifier(classifier, X, y):
    x_min, x_max = min(X[:, 0]) - 0.2, max(X[:, 0]) + 0.2 # 计算图中坐标的范围
    y_min, y_max = min(X[:, 1]) - 0.2, max(X[:, 1]) + 0.2
    step_size = 0.01 # 设置step size
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
                                     np.arange(y_min, y_max, step_size))
    # 构建网格数据
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black',
                linewidth=0.5, cmap=plt.cm.Paired)
    # specify the boundaries of the figure
    #plt.xlim(x_values.min(), x_values.max())
    #plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
   # plt.xticks((np.arange(int(min(X[:, 0])), int(max(X[:, 0])), 0.5)))
    #plt.yticks((np.arange(int(min(X[:, 1])), int(max(X[:, 1])), 0.5)))
    plt.grid()
    plt.show()

# 模型结果可视化
plot_classifier( log_reg,X1,y1)
log_reg_pred = log_reg.predict(X1)
target_names = ['Class-0', 'Class-1']
print(classification_report(y1, log_reg_pred, target_names=target_names))
