import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# 将数据集分成训练集和测试集
x = loadmat("./data_train.mat")['data_train']
y = loadmat("./label_train.mat")['label_train']
data_test = loadmat("D:\py_projects\EE7207_Assignments\data_test.mat")['data_test']
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

# 设置gridsearch的参数
tuned_parameters = {'gamma': np.linspace(0.1, 1, 19),
                    'C': np.linspace(0.1, 1, 19)}
print("parameters:", format(tuned_parameters))

grid_search = GridSearchCV(SVC(), tuned_parameters)
grid_search.fit(X_train, y_train.ravel())

print("best scores on training set: ", format(grid_search.best_score_))
print("best parameters: ", format(grid_search.best_params_))
print("test set score: ", format(grid_search.score(X_test, y_test)))
print("prediction: ", format(grid_search.predict(data_test)))
