# -*- coding: utf-8 -*-
"""
作业13
张雅洁 2016111033

use CV to select tuning parameter in ridge regression
"""

import numpy as np
from sklearn.model_selection import KFold
import numpy.linalg as la
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
 
##homework 
#构建函数
def cv_ridge_ld(n,ld_seq,x,y):
    #k_fold
    z = np.hstack((x,y))
    kf = KFold(n_splits=n)
    train_seq = []
    test_seq = []
    for train, test in kf.split(z):
        train_seq.append(train)
        test_seq.append(test)
    #cv
    cv_seq = []
    for ld in ld_seq:
        cv = 0
        for j in range(n):
            x_train = x[train_seq[j]]
            y_train = y[train_seq[j]]
            x_test = x[test_seq[j]]
            y_test = y[test_seq[j]]
            ridge_reg = Ridge(alpha=ld, solver='sag')
            ridge_reg.fit(x_train,y_train)
            yhat=ridge_reg.predict(x_test)
            cv = cv + np.mean((y_test - yhat)**2)/n
        cv_seq.append(cv)
    return cv_seq


##Example
n = 100
x = np.random.rand(n,4)
beta_true = np.array([1,2,3,-4])
err = np.random.randn(n,1)*0.3
y = x.dot(beta_true.reshape(4,1)) + err

ld_seq =  np.linspace(0.001,1,100)

cv_seq=cv_ridge_ld(4,ld_seq,x,y)
id=np.argmin(cv_seq)
ld=ld_seq[id]##0.011090909090909092
print(ld)
plt.plot(ld_seq,cv_seq)






