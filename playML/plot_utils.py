# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, axis):
    """绘制不规则决策边界"""
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(1, -1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(1, -1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)

def plot_svc_decision_boundary(model, axis):
    plot_decision_boundary(model, axis)
    w = model.coef_[0]
    b = model.intercept_[0]

    # 绘制margin的直线
    # 决策边界所在直线的表达式：w0 * x0 + w1 * x1 + b = 0  -> x1 = -w0 * x0 / w1 - b / w1
    plot_x = np.linspace(axis[0], axis[1], 200)

    # w0 * x0 + w1 * x1 + b = 1  -> x1 = 1/w1 - w0 * x0 / w1 - b / w1
    up_y = -w[0]/w[1]*plot_x - b/w[1] + 1/w[1]

    down_y = -w[0]/w[1]*plot_x - b/w[1] - 1/w[1]

    # 处理超过了坐标轴范围的值
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])

    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')