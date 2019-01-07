#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

A = 150
B = 90
R = 15
h = 5
P = 80
E = 50
v = 0.33
D = E * h ** 3 / (12 - (1 - v ** 2))

def build_grid_norm(A, B, P, D, R, h, eps = 0.1):
    matrix = np.zeros((A + 1, B + 1))
    current_delta = 1
    count = 0
    while current_delta > eps:
        current_delta = 0
        for i in range(1, A):
            for j in range(1, B):
                if (A / 2 - i) ** 2 + (0 - j) ** 2 <= R ** 2:
                    continue
                temp = matrix[i][j]
                matrix[i][j] = 0.25 * (
                            matrix[i - 1][j] + matrix[i + 1][j] + matrix[i][j - 1] + matrix[i][j + 1] - h ** 2 * P / D)
                delta = abs(temp - matrix[i][j])
                if delta > current_delta:
                    current_delta = delta
        count += 1
        
    print(count)
    return matrix


result = build_grid_norm(A, B, P, D, R, h)
fig = plt.figure()
ax = Axes3D(fig)
u_x = np.linspace(0, 1, B + 1)
u_y = np.linspace(0, 1, A + 1)
X, Y = np.meshgrid(u_x, u_y)
Z = result
ax.plot_wireframe(X, Y, Z)
plt.show()

