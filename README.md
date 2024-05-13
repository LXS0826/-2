# Iris Species Classification using Logistic Regression

这个项目展示了如何使用逻辑回归模型对鸢尾花数据集进行分类。我们通过 scikit-learn 的 LogisticRegression 类来实现对鸢尾花种类的预测。

## 安装指南

在运行这个项目之前，你需要安装以下 Python 包：

- scikit-learn
- pandas
- numpy
- matplotlib

你可以通过以下命令来安装这些包：

```bash
pip install scikit-learn pandas numpy matplotlib
数据集
我们使用了 scikit-learn 库中内置的鸢尾花数据集。这个数据集包含了150个样本，分别属于三种不同的鸢尾花类别。每个样本都有四个特征，但在我们的逻辑回归模型中，我们只使用了其中的两个特征（花瓣长度和宽度）。

使用方法
克隆这个仓库到你的本地机器上：

git clone https://github.com/your-username/iris-logistic-regression.git
cd iris-logistic-regression
运行 script.py 文件：

python script.py
结果可视化
该脚本会输出逻辑回归模型的参数，并在一个图中展示模型的分类效果。图中会显示数据点和决策边界，以及模型的准确率。
