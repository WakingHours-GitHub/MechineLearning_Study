from sklearn.datasets import load_iris
from sklearn.model_selection import  train_test_split

def datasets_demo():
    """
    sklearn数据集的使用
    :return: None
    """
    iris = load_iris()
    print("鸢尾花数据集:", iris)
    print("查看数据集描述:", iris["DESCR"])
    print("查看特征值的名字:", iris.feature_names)
    print("查看样本个数: ", iris.data, iris.data.shape)

    # 数据集的划分:
    # sklearn中的API: 帮助我们直接分开训练集和数据集
    # sklearn.model_selection.train_test_split(arrays, *options)
    # x数据集的特征值, y数据集的特征值
    # test_size: 测试集大小, 一般为float, 例如0.2
    # random_state随机数种子, 不同的种子会造成不同的随机效果.相同的种子采样相同
    # return:  训练集特征值，测试集特征值，训练集目标值，测试集目标值
    # 我们统计: x_train, x_test, y_train, y_test
    # 我自己的命名方式: train_feature, test_feature, train_target, test_target

    train_feature, test_ferture, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练的数据集: ", train_feature, train_target)

    return None


if __name__ == '__main__':
    datasets_demo()
