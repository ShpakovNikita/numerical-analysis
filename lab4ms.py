#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sympy
from sympy.solvers.solveset import linsolve
from sympy.solvers import solve
import matplotlib.pyplot as plt
from copy import copy

from mpl_toolkits.mplot3d import Axes3D
from numpy import linspace
import numpy

x = sympy.Symbol('x')
t = sympy.Symbol('t')

def calc_func_kx(matrix, i, j, func, h, tau, k):
    val = (func.subs([(t, tau * (i - 1)), (x, h * (j))])) * tau\
           + matrix[i - 1, j] \
           + tau * ( sympy.diff(k, x).subs(x, h * (j)) * (matrix[i - 1, j] - matrix[i - 1, j - 1]) / h \
           + k.subs(x, h * (j)) * (matrix[i - 1, j - 1] - 2 * matrix[i - 1, j] + matrix[i - 1, j + 1]) / h ** 2)
    return val


def build_grid(func, start, a, b, y_a, y_b, h, tau, k=(1 + 0 * x), T=None, 
               calc=calc_func_kx):
    n = int((b - a) / h)
    if not T:
        n_t = n
    else:
        n_t = int(T / tau)

    matrix = numpy.zeros((n_t + 1, n + 1))

    for i in range(n_t):
        matrix[0, i] = start.subs(x, h * i)

    for i in range(n + 1):
        matrix[i, 0] = y_a.subs(t, 0)
        matrix[i, n_t] = y_b.subs(t, tau * (n_t - 1))

    for i in range(1, n_t):
        for j in range(1, n):
            matrix[i, j] = calc(matrix, i, j, func, h, tau, k)

    return matrix

# TASK 1
print("TASK 1")
# just clamps the value
def clamp(val, mi_, ma_):
    return max(min(val, ma_), mi_)

k = 1
# redefine
k = 1

T = 0.2
a = -1
b = 1
g_1 = 1 + 0 * t
g_2 = 5 * t
num = 10

h = (b - a) / num
tau = clamp(T / num, 0.001, (a - b) ** 2 / (200 * k))
true_count = int(T / tau)

data = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'burlywood', 'chartreuse']

X = linspace(a, b, num + 1)

func_1 = 0 * x
func_start_1 = 1 -  x ** 2

res_matrix = build_grid(func_1, func_start_1, a, b, g_1, g_2, h, tau, T=T)

plt.figure()
for i in range(num):
    plt.plot(X, res_matrix[i], color=colors[-i])
    
plt.title('task_4: ')
plt.grid(True)
plt.show()

for i in range(num):
    plt.figure()
    plt.plot(X, res_matrix[i])   
    plt.title('task_4: ')
    plt.grid(True)
    plt.show()

u = linspace(a, b, num + 1)
y = linspace(0, (true_count + 1) * tau, (true_count + 1))
X, Y = numpy.meshgrid(u, y)

z_ = []
for i in range(true_count + 1):
    z_ += res_matrix[i].tolist()
    
zs = numpy.array(z_)
Z = zs.reshape(X.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
