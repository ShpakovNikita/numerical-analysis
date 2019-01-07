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
import functools
from collections import namedtuple


POWER = 1
ITER_COUNT = 20
LINSPACE_SIZE = 20
x = sympy.Symbol('x')
t = sympy.Symbol('t')


def k_1(x, c):
    return 1 / (sympy.cos(x) ** 2)


def function_1(x):
    return 6 * (sympy.cos(x) ** 3)

def function_2(x):
    return x

# this function build's u(x) by double integration of given equation
def build_final_equation(f, k, c, c1, c2):
    first_iter = ((-sympy.integrate(f(x), x) + c1) / k(x, c)).expand()
    second_iter = sympy.integrate(first_iter, x) + c2
    return second_iter


def func_for_partition_2(yk_m1, yk, yk_p1, h, k=1, func=None):
    if not func:
        func = function_1
        
    func = -k * (yk_p1 - 2 * yk + yk_m1) / h ** 2 - func(x)
    return func

def func_for_partition_3(yk_m1, yk, yk_p1, h, tau, t, k=1, func=None):
    if not func:
        func = function_1
        
    func = ((yk_p1 - 2 * yk + yk_m1) / h ** 2 + 2 * sympy.tan(x) * \
            (yk_p1 - yk_m1) / (2 * h)) / (sympy.cos(x) ** 2) - func(x) * \
            (1 - sympy.exp(-t)) - (yk_p1 - yk_m1) / (2 * tau)
    return func

def func_for_partition_4(yk_m1, yk, yk_p1, h, tau, t, k=1, func=None):
    if not func:
        func = function_2
        
    func = k * (yk_p1 - 2 * yk + yk_m1) / h ** 2 + func(x) -  (yk_p1 - yk_m1)\
    / (2 * tau)
    return func

# this function returns final thermal conductivity function
def solve_thermal_conductivity_equation(f, k, c, a, b, u_a, u_b):
    c1, c2 = sympy.Symbol('c1'), sympy.Symbol('c2')
    true_eq = build_final_equation(f, k, c, c1, c2)
    c2_val = solve(true_eq.subs(x, b) - u_b, c2)[0]
    true_eq = true_eq.subs(c2, c2_val)
    
    c1_val = solve(true_eq.subs(x, a) - u_a, c1)[0]
    true_eq = true_eq.subs(c1, c1_val)
    return true_eq


# differences method like in 2rd lab work. points_k is a list of tuples with 
# proper k to the point from prev point, ex: [(point_until, k1), (...), ...]
def differences_method(start_variables_count, 
                       a,
                       b, 
                       y_a, 
                       y_b, 
                       func_for_partition,
                       points_k,
                       func=None):

    # generate variables for linsolve
    start_variables_count += 2
    symbols = [sympy.Symbol('y' + str(i)) for i in
               range(start_variables_count)]
    # define our step
    h = (b - a) / start_variables_count 
    # generate x points:
    points = linspace(a + h, b - h, start_variables_count - 2).tolist()
    # build system of equations
    lin_system = []
    selected_k = 0
    for i in range(1, start_variables_count - 1):
        if points[i - 1] > points_k[selected_k][0]:
            selected_k += 1
            
        lin_system.append(func_for_partition(symbols[i - 1], 
                                             symbols[i], 
                                             symbols[i + 1],
                                             h,
                                             points_k[selected_k][1],
                                             func).evalf())
    for i in range(start_variables_count - 2):
        lin_system[i] = lin_system[i].subs(x, points[i])
    
    lin_system[0] = lin_system[0].subs(symbols[0], y_a)
    lin_system[-1] = lin_system[-1].subs(symbols[-1], y_b)
    
    del symbols[0], symbols[-1]
    
    # solving our system and converting FiniteSet to simple list
    answer = list(list(linsolve(lin_system, *symbols))[0])
    
    data_type = namedtuple('data', 
                          ('points', 'answer', 'step'))
    points.insert(0, a)
    points.append(b)
    
    answer.insert(0, y_a)
    answer.append(y_b)
    
    return data_type(points, answer, (b - a) / start_variables_count)

# TASK 1
print("TASK 1")

a = 0.5
b = 1.3
u_a = 2
u_b = 2

var_1 = solve_thermal_conductivity_equation(function_1, k_1, 1, a, b, u_a, u_b)
var_2 = solve_thermal_conductivity_equation(function_1,
                                            lambda x, c: c * k_1(x, c),
                                            2, a, b, u_a, u_b)
var_3 = solve_thermal_conductivity_equation(function_1, 
                                            lambda x, c: c * k_1(x, c),
                                            0.1, a, b, u_a, u_b)
print(var_1)

D = linspace(a, b, LINSPACE_SIZE)
func_y1, func_y2, func_y3, func_y4 = [], [], [], []
for i in range(len(D)):
    func_y1.append(var_1.subs(x, D[i]))
    func_y2.append(var_2.subs(x, D[i]))
    func_y3.append(var_3.subs(x, D[i]))
    

fig, _ = plt.subplots()
plt.plot(D, func_y1, c = 'red', label = 'v1')
plt.plot(D, func_y2, c = 'green', label = 'v2')
plt.plot(D, func_y3, c = 'blue', label = 'v3')

plt.legend()
plt.show()

var_4 = solve_thermal_conductivity_equation(function_1, 
                                            lambda x, c: 1 / k_1(x, c),
                                            1, a, b, u_a, u_b)

for i in range(len(D)):
    func_y4.append(var_4.subs(x, D[i]))

fig, _ = plt.subplots()
plt.plot(D, func_y1, c = 'red', label = 'v1')
plt.plot(D, func_y4, c = 'green', label = 'v4')

plt.legend()
plt.show()

var_5 = solve_thermal_conductivity_equation(function_1, k_1, 1, a, b, -u_a, 
                                            u_b)
var_6 = solve_thermal_conductivity_equation(function_1, k_1, 1, a, b, u_a, 
                                            -u_b)
var_7 = solve_thermal_conductivity_equation(function_1, k_1, 1, a, b, -u_a, 
                                            -u_b)

func_y5, func_y6, func_y7 = [], [], []
for i in range(len(D)):
    func_y5.append(var_5.subs(x, D[i]))
    func_y6.append(var_6.subs(x, D[i]))
    func_y7.append(var_7.subs(x, D[i]))

fig, _ = plt.subplots()
plt.plot(D, func_y5, c = 'red', label = 'v5')
plt.plot(D, func_y6, c = 'green', label = 'v6')
plt.plot(D, func_y7, c = 'blue', label = 'v7')

plt.legend()
plt.show()

# TASK 2
print("TASK 2")
# a
data_a1 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(0.5 * (b + a), 0.01), (b, 100)])
data_a2 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(0.5 * (b + a), 100), (b, 0.01)])

D1, y1 = data_a1.points, data_a1.answer
D2, y2 = data_a2.points, data_a2.answer
plt.figure()
plt.plot(D1, y1, color='red')
plt.plot(D2, y2)
plt.title('a task_2: ')
plt.grid(True)
plt.show()

# b
data_b1 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(a + 1/3 * (b - a), 0.1),
                              (a + 2/3 * (b - a), 0.2),
                              (b, 0.3)])
data_b2 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(a + 1/3 * (b - a), 0.3),
                              (a + 2/3 * (b - a), 0.2),
                              (b, 0.1)])
data_b3 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(a + 1/3 * (b - a), 0.1),
                              (a + 2/3 * (b - a), 0.2),
                              (b, 0.1)])
data_b4 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(a + 1/3 * (b - a), 2),
                              (a + 2/3 * (b - a), 0.1),
                              (b, 2)])
        
D1, y1 = data_b1.points, data_b1.answer
D2, y2 = data_b2.points, data_b2.answer
D3, y3 = data_b3.points, data_b3.answer
D4, y4 = data_b4.points, data_b4.answer
plt.figure()
plt.plot(D1, y1, color='red')
plt.plot(D2, y2, color='green')
plt.plot(D3, y3, color='blue')
plt.plot(D4, y4, color='yellow')
plt.title('b task_2: ')
plt.grid(True)
plt.show()

# c
data_c1 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(b, 1)], 
                             lambda x: POWER * (x - (a + (b - a) * 0.5)))
data_c2 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(b, 1)], 
                             lambda x: POWER * (x - (a + (b - a) * 0.3)) * \
                             POWER * (x - (a + (b - a) * 0.7)))
data_c3 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(b, 1)], 
                             lambda x: POWER * (x - (a + (b - a) * 0.3)) * \
                             POWER * 5 * (x - (a + (b - a) * 0.7)))
data_c4 = differences_method(ITER_COUNT, a, b, u_a, u_b, func_for_partition_2, 
                             [(b, 1)], 
                             lambda x: POWER * (x - (a + (b - a) * 0.5)) * \
                             POWER * (x - (a + (b - a) * 0.7)))

D1, y1 = data_c1.points, data_c1.answer
D2, y2 = data_c2.points, data_c2.answer
D3, y3 = data_c3.points, data_c3.answer
D4, y4 = data_c4.points, data_c4.answer
plt.figure()
plt.plot(D1, y1, color='red')
plt.plot(D2, y2, color='green')
plt.plot(D3, y3, color='yellow')
plt.plot(D4, y4)
plt.title('c task_2: ')
plt.grid(True)
plt.show()

# TASK 3
print("TASK 3")
def calc_func(matrix, i, j, func, h, tau, k=1):
    val = (matrix[i - 1, j - 1] * tau / h ** 2 \
           + (1 - 2 * tau / h ** 2) * matrix[i - 1, j] \
           + tau / h ** 2 * matrix[i - 1, j + 1] + tau * func.
           subs([(t, tau * (i - 1)), (x, h * (j))]))
    return val

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
        matrix[i, 0] = y_a
        matrix[i, n_t] = y_b

    for i in range(1, n_t):
        for j in range(1, n):
            matrix[i, j] = calc(matrix, i, j, func, h, tau, k)
            
    return matrix


def differences_method_3(d_tau,
                         d_h,
                         a,
                         b, 
                         y_a, 
                         y_b, 
                         func_for_partition,
                         t_func,
                         k=1,
                         func=None):

    start_variables_count = int((b - a) / d_h)
    # generate variables for linsolve
    symbols = [sympy.Symbol('y' + str(i)) for i in \
               range(start_variables_count)]
    # generate x points:
    points = linspace(a + d_h, b - d_h, start_variables_count - 2).tolist()
    # build system of equations
    lin_system = []
    for i in range(1, start_variables_count - 1):
        lin_system.append(func_for_partition(symbols[i - 1], 
                                             symbols[i], 
                                             symbols[i + 1],
                                             d_h,
                                             d_tau,
                                             t_func(d_tau),
                                             k,
                                             func).evalf())
    for i in range(start_variables_count - 2):
        lin_system[i] = lin_system[i].subs(x, points[i])
    
    lin_system[0] = lin_system[0].subs(symbols[0], y_a)
    lin_system[-1] = lin_system[-1].subs(symbols[-1], y_b)
    
    del symbols[0], symbols[-1]
    
    # solving our system and converting FiniteSet to simple list
    answer = list(list(linsolve(lin_system, *symbols))[0])
    
    points.insert(0, a)
    points.append(b)
    
    answer.insert(0, y_a)
    answer.append(y_b)
    
    return points, answer

tau = 0.05
h = 0.01

k = 1 / (sympy.cos(x)) ** 2
a = 0.5
b = 1.3
u_a = 2
u_b = 2

func_start_3 = 2 + 0 * x;
func_3 = 6 * (sympy.cos(x)) ** 3 * (1 - sympy.exp(-t))

res_matrix = build_grid(func_3, func_start_3, a, b, u_a, u_b, h, tau, k=k)

n = int((b - a) / h)
u = linspace(a, b, n + 1)
y = linspace(0, (n + 1) * tau, n + 1)
X, Y = numpy.meshgrid(u, y)

z_ = []
for i in range(n + 1):
    z_ += res_matrix[i].tolist()
    
zs = numpy.array(z_)
Z = zs.reshape(X.shape)

t_0 = res_matrix[0]
t_5 = res_matrix[5]
t_20 = res_matrix[20]
t_70 = res_matrix[70]

plt.plot(u, t_0)
plt.title('t = 0')
plt.grid()
plt.show()

plt.plot(u, t_5)
plt.title('t = 5 tay')
plt.grid()
plt.show()

plt.plot(u, t_20)
plt.title('t = 20 tay')
plt.grid()
plt.show()

plt.plot(u, t_70)
plt.title('t = 70 tay')
plt.grid()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

"""
D1, y1 = differences_method_3(tau, h, a, b, u_a, u_b, func_for_partition_3, 
                             lambda x: 0 )
D2, y2 = differences_method_3(tau, h, a, b, u_a, u_b, func_for_partition_3, 
                             lambda x: 5 * x )
D3, y3 = differences_method_3(tau, h, a, b, u_a, u_b, func_for_partition_3, 
                             lambda x: 20 * x )
D4, y4 = differences_method_3(tau, h, a, b, u_a, u_b, func_for_partition_3, 
                             lambda x: 200 * x )

plt.figure()
plt.plot(D1, y1, color='red')
plt.plot(D2, y2, color='green')
plt.plot(D3, y3, color='blue')
plt.plot(D4, y4, color='yellow')
plt.title('task_3: ')
plt.grid(True)
plt.show()
"""

# TASK 4
print("TASK 4")
# just clamps the value
def clamp(val, mi_, ma_):
    return max(min(val, ma_), mi_)

k = 0.5 + 0 * x

T = 0.4
a = -1
b = 1
g_1 = 1
g_2 = 1
num = 10

h = (b - a) / num
tau = clamp(T / num, 0.001, (a - b) ** 2 / (200 * k))
true_count = int(T / tau)

data = []
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'burlywood', 'chartreuse']

X = linspace(a, b, num + 1)

func_4 = x
func_start_4 = x ** 2

res_matrix = build_grid(func_4, func_start_4, a, b, g_1, g_2, h, tau, T=T, k=k)

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

"""
for i in range(true_count):
    data.append(differences_method_3(tau, h, a, b, g_1, g_2(i * 2 * tau), 
                                     func_for_partition_4, 
                                     lambda x: i * 2 * x, k ))

plt.figure()
for i in range(true_count):
    plt.plot(data[i][0], data[i][1], color=colors[-i])
    
plt.title('task_4: ')
plt.grid(True)
plt.show()
"""
