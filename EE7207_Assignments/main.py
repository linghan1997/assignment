import numpy as np
from matplotlib import pyplot as plt
from numpy import argmax
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

from rbf import RBF

np.random.seed(0)

def calc_accuracy(Y, Y_g):
    res = len(np.where((Y - Y_g) == 0)[0]) / len(Y_g)
    return res


if __name__ == '__main__':
    data_test = loadmat("D:\py_projects\EE7207_Assignments\data_test.mat")
    data_train = loadmat("D:\py_projects\EE7207_Assignments\data_train.mat")
    label_train = loadmat("D:\py_projects\EE7207_Assignments\label_train.mat")

    x = data_train['data_train']
    y = label_train['label_train']
    x_test = data_test['data_test']

    # 划分数据集与验证集，row = #features
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)

    # accuracy: 4 methods, 2 for training & validation, 50 choices of #neurons
    accuracy = np.zeros((4, 2, 35))

    # 调整#neurons = m [2, 36]
    m = 35
    n = x_train.shape[1]
    # RBF(a, b, c): a = #features, b = #neurons, c = #outputs
    # method 1
    for i in range(m):
        rbf1 = RBF(n, 2 + i, 1)
        rbf1.train_RS(x_train, y_train)
        y_train_predict = rbf1.predict(x_train)
        y_val_predict = rbf1.predict(x_val)
        accuracy[0, 0, i] = rbf1.cal_accuracy(y_train_predict, y_train)
        accuracy[0, 1, i] = rbf1.cal_accuracy(y_val_predict, y_val)

    print("method1 accomplished")

    # method 2
    for i in range(m):
        rbf2 = RBF(n, 2 + i, 1)
        rbf2.train_PT(x_train, y_train)
        y_train_predict = rbf2.predict(x_train)
        y_val_predict = rbf2.predict(x_val)
        accuracy[1, 0, i] = rbf2.cal_accuracy(y_train_predict, y_train)
        accuracy[1, 1, i] = rbf2.cal_accuracy(y_val_predict, y_val)

    print("method2 accomplished")

    # # method 3
    # for i in range(m):
    #     rbf3 = RBF(n, 2 + i, 1)
    #     rbf3.train_CS(x_train, y_train)
    #     y_train_predict = rbf3.predict(x_train)
    #     y_val_predict = rbf3.predict(x_val)
    #     accuracy[2, 0, i] = rbf3.cal_accuracy(y_train_predict, y_train)
    #     accuracy[2, 1, i] = rbf3.cal_accuracy(y_val_predict, y_val)
    #
    # print("method3 accomplished")
    #
    # # method 4
    # for i in range(m):
    #     rbf4 = RBF(n, 2 + i, 1)
    #     rbf4.train_BP(x_train, y_train, 100, 0.5, 0.5, 0.5)
    #     y_train_predict = rbf4.predict(x_train)
    #     y_val_predict = rbf4.predict(x_val)
    #     accuracy[3, 0, i] = rbf4.cal_accuracy(y_train_predict, y_train)
    #     accuracy[3, 1, i] = rbf4.cal_accuracy(y_val_predict, y_val)
    #     print("train_num:", i + 1)
    #     print("train:", accuracy[3, 0, i])
    #     print("valid:", accuracy[3, 1, i])
    #
    # print("method4 accomplished")

    x = np.linspace(2, 36, 35)
    # training accuracy
    plt.figure(1)
    plt.plot(x, accuracy[0, 0, :], marker='o', color='red')
    plt.plot(x, accuracy[1, 0, :], marker='x', color='blue')
    plt.plot(x, accuracy[2, 0, :], marker='^', color='purple')
    # plt.plot(x, accuracy[3, 0, :], marker='v', color='orange')
    plt.grid()
    plt.ylim([0, 1])
    plt.title("Training Accuracy")
    plt.xlabel("#Neurons")
    plt.ylabel("Accuracy")
    plt.legend(['Method 1', 'Method 2', 'Method 3'], loc="best")
    plt.show()

    # validation accuracy
    plt.figure(2)
    plt.plot(x, accuracy[0, 1, :], marker='o', color='red')
    plt.plot(x, accuracy[1, 1, :], marker='x', color='blue')
    plt.plot(x, accuracy[2, 1, :], marker='^', color='purple')
    # plt.plot(x, accuracy[3, 1, :], marker='v', color='orange')
    plt.grid()
    plt.ylim([0, 1])
    plt.title("Validation Accuracy")
    plt.xlabel("#Neurons")
    plt.ylabel("Accuracy")
    plt.legend(['Method 1', 'Method 2', 'Method 3'], loc="best")
    plt.show()

    # Predict
    # rbf_op = RBF(n, 32, 1)
    # rbf_op.train_CS(x_train, y_train)
    # y_test_predict = rbf_op.predict(x_test)
    # print("outcome: ", y_test_predict)

    idx = argmax(accuracy[1, 1, :])
    print("valid accuracy: ", accuracy[1, 1, idx])
    print("best #neurons: ", idx + 2)
    rbf_op = RBF(n, idx + 2, 1)
    rbf_op.train_PT(x_train, y_train)
    y_test_predict = rbf_op.predict(x_test)
    print("prediction: ", y_test_predict)
