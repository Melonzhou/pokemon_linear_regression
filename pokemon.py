#!/usr/bin/python
# -*- coding: UTF-8 -*-
import matplotlib

matplotlib.use('TkAgg')
# 2,3行的代码不能省略，否则matplotlib导入会报错

import matplotlib.pyplot as plt
import numpy as np

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]

x = np.arange(-200, -100, 1)
y = np.arange(-5, 5, 0.1)
Z = np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)

for i in range(len(x)):
    for j in range(len(y)):
        b = x[i]
        w = y[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (y_data[n] - b - w * x_data[n]) ** 2
        Z[j][i] = Z[j][i] / len(x_data)

# y_output = b + w * x_data 假设我们的模型是这样的
# L(w,b) = 累加[(y_data - b - w * x_data)**2], 损失函数


b = -120  # 随意给b，w取值
w = -4
lr = 1
iteration = 100000

b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for m in range(0, 10000):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        # 计算在b和w上的梯度，向梯度下降的地方走
        # 对b和w求偏导数，然后累加
        # L对b的偏导为 累加[2.0 * (y_data - b - w * x_data) * (-1)]
        # L对w的偏导为 累加[2.0 * (y_data - b - w * x_data) * (-)x_data]

        b_grad = b_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * 1.0
        w_grad = w_grad - 2.0 * (y_data[n] - b - w * x_data[n]) * x_data[n]

    # 更新参数w,b
    # b = b - lr * b_grad
    # w = w - lr * w_grad

    # 更新参数w,b，这是大招, adagrad
    lr_b = lr_b + b_grad ** 2
    lr_w = lr_w + w_grad ** 2

    b = b - lr / np.sqrt(lr_b) * b_grad
    w = w - lr / np.sqrt(lr_w) * w_grad

    # 保存每一步计算得到的w，b，用来画图
    b_history.append(b)
    w_history.append(w)

# 画出gradient descent的路径
plt.contourf(x, y, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot([-188.4], [2.67], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5, color='black')
plt.xlim(-200, -100)
plt.ylim(-5, 5)
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
