import sympy
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PI = sympy.pi
a = 2
b = 2
T = 2

h = 0.2
tau = 0.5 * h ** 2

C = sympy.Symbol('C')
t = sympy.Symbol('t')
x = sympy.Symbol('x')
y = sympy.Symbol('y')

d_y = y

C_subs = 1

u_0 = 2 * sympy.cos(PI * x / a)
u_t_0 = sympy.tan(sympy.sin(2 * PI * x / a)) * sympy.sin(PI  * y / b)

u = - t ** 2 / 2 * C + t * u_t_0 + u_0

def gen_points_mat(a, b, h):
    mat = []
    count_a = int(a / h) 
    count_b = int(b / h)
    
    start_y = - b / 2 
    start_x = - a / 2
    
    for i in range(count_b):
        row = []
        for j in range(count_a):
            row.append((start_x + j * h, start_y + i * h))
        
        mat.append(row)
    
    
    return mat, count_a, count_b


def gen_u_mat(points, res_func):
    mat = []
    
    for i in range(len(points)):
        row = []
        for j in range(len(points[0])):
            row.append(res_func.subs([(x, points[i][j][0]), (y, points[i][j][1])]))
        
        mat.append(row)
    
    for i in range(len(points)):
        mat[i][0] = 0
        mat[i][-1] = 0
    
    for j in range(len(points[0])):
        mat[0][j] = 0
        mat[-1][j] = 0

    return mat

#TASK 2 (grids)
def func_factory(mat, i, j):
    global u_0, u_t_0, tau, h
    res = u_0 + tau * u_t_0
    res += tau ** 2 / 2 * ((mat[i - 1][j] - 2 * mat[i][j] + mat[i + 1][j]) / h ** 2 + (mat[i][j - 1] - 2 * mat[i][j] + mat[i][j + 1]) / h ** 2)
    return res
    
def gen_first_layer(points, func):
    return gen_u_mat(points, func)


def gen_second_layer(points, layer_0, func):
    mat = [[0 for i in range(len(points[0]))]]
    
    for i in range(1, len(points) - 1):
        row = [0]
        for j in range(1, len(points[0]) - 1):
            row.append(func(layer_0, i, j).subs([(x, points[i][j][0]), (y, points[i][j][1])]))
        row.append(0)
        mat.append(row)

    mat.append([0 for i in range(len(points[0]))])
    return mat


def build_volume_grid(points, layer_0, layer_1, T, h_t, h, func):
    count_t = int(T / h_t)
    
    volume = np.zeros((count_t, len(points), len(points[0])))
    
    volume[0] = layer_0
    volume[1] = layer_1
    
    for tau in range(count_t):
        for j in range(len(points[0])):
            volume[tau][0][j] = 0
            volume[tau][-1][j] = 0
    
    for tau in range(count_t):
        for i in range(len(points[0])):
            volume[tau][i][0] = 0
            volume[tau][i][-1] = 0
            
    
    for tau in range(2, count_t):
        for i in range(1, len(points) - 1):    
            for j in range(1, len(points[0]) - 1):
                volume[tau][i][j] = func(volume, tau, i, j, h, h_t)     
    
    return volume, count_t


def grid_func(volume, t, y, x, h, h_t):
    res = volume[t - 1][y - 1][x] - 4 * volume[t - 1][y][x] + volume[t - 1][y + 1][x] + volume[t - 1][y][x - 1] + volume[t - 1][y][x + 1]
    res = res / h ** 2 * h_t ** 2
    res -= (volume[t - 2][y][x] - 2 * volume[t - 1][y][x])
    return res

#create first layer
points, count_x, count_y = gen_points_mat(a, b, h)
layer_0 = gen_first_layer(points, u_0)

#create second layer
layer_1 = gen_second_layer(points, layer_0, func_factory)

volume, count_t = build_volume_grid(points, layer_0, layer_1, T, tau, h, grid_func)

a = np.linspace(-a / 2, a / 2, count_x)
b = np.linspace(-b / 2, b / 2, count_y)
X, Y = np.meshgrid(a, b)

for t_c in range(count_t):
    fig = plt.figure()
    ax = Axes3D(fig)
    Z = volume[t_c]
    ax.plot_wireframe(X, Y, Z)
    plt.show()


#TASK 1
def func_task_1(mat, tau, i, p, E, h_t, h):
    res = (mat[tau - 1][i - 1] - 2 * mat[tau - 1][i] + mat[tau - 1][i + 1]) / h ** 2
    res *= (h_t ** 2)
    res += (2 * mat[tau - 1][i] - mat[tau - 2][i])
    return res    


def build_plot_data(h_t, h, T, L, du, p, E, func):
    count_t = int(T / h_t)
    count_h = int(L / h)
    
    mat = np.zeros((count_t, count_h))
    
    # first layer
    for i in range(count_h):
        if i * h <= L / 2:
            mat[0][i] = 2 * du * i * h / L
        else:
            mat[0][i] = 2 * du * (1 - i * h / L)
    
    # second layer
    for i in range(1, count_h - 1):
        mat[1][i] = mat[0][i] - h_t ** 2 / 2 * (mat[0][i - 1] - 2 * mat[0][i] + mat[0][i + 1]) / h ** 2
    
    for tau in range(count_t):
        mat[tau][0] = 0
        mat[tau][-1] = 0
    
    for tau in range(2, count_t):
        for i in range(1, count_h - 1):
            mat[tau][i] = func(mat, tau, i, p, E, h_t, h)
    
    return count_t, count_h, mat

# redefine steps and total time
T = 5
h = 0.3
tau = 0.5 * h ** 2

L = 6
du = 0.15
E = 82 * (10 ** 9)
p = 9.7 * (10 ** 3)

count_t, count_h, solution = build_plot_data(tau, h, T, L, du, p, E, func_task_1)
x_data = np.arange(0, L, h)

for t_c in range(count_t):
    plt.figure()
    plt.plot(x_data, solution[t_c])   
    plt.title('time: ' + str(t_c * tau))
    plt.grid(True)
    plt.show()

    
    