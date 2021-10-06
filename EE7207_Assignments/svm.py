from numpy import argmax
from sklearn import svm, model_selection
from scipy.io import loadmat
import numpy as np
from matplotlib import pyplot as plt

# np.random.seed(0)

x = loadmat("./data_train.mat")['data_train']
y = loadmat("./label_train.mat")['label_train']
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=0.3)
x_test = loadmat("./data_test.mat")['data_test']

print("data size:", x.shape)

train_score = []
validation_score = []
svm_obj = []
list = []
# gamma ranges from 0.1~1
for i in range(20):
    gamma = 0.1 + 0.05 * i
    gamma = round(gamma, 5)
    clf = svm.SVC(C=0.7, kernel='rbf', gamma=gamma, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    svm_obj.append(clf)
    train_score.append(clf.score(x_train, y_train))
    validation_score.append(clf.score(x_validation, y_validation))
    list.append(gamma)

clf = svm.SVC(C=0.7, kernel='rbf', gamma="scale", decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())
train_score_auto = clf.score(x_train, y_train)
validation_auto = clf.score(x_validation, y_validation)

x = np.linspace(0.1, 1, 20)
plt.plot(x, train_score, x, validation_score, marker="v")
plt.scatter(1 / x_train.shape[0], train_score_auto, marker="o")
plt.scatter(1 / x_train.shape[0], validation_auto, marker="o")
plt.grid()
plt.title("K-SVM Accuracy")
plt.xlabel("Gamma")
plt.ylabel("Accuracy")
plt.ylim([0.8, 1])
plt.legend(['training set', 'validation set', 'training set: gamma = scale',
            'validation set: gamma = scale'], loc="best")
plt.show()
# plt.savefig('svm.jpg')

# find the optimization
idx = argmax(validation_score)
print("best validation accuracy: ", validation_score[idx])
print("best gamma: ", list[idx])
y_test = svm_obj[idx].predict(x_test)
print("prediction: ", y_test)
