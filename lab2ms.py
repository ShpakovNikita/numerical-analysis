#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sympy
from sympy.solvers.solveset import linsolve

from collections import namedtuple
from numpy import linspace
import matplotlib.pyplot as plt


x = sympy.Symbol('x')

# ---------------- DATA FOR MODIFYING -----------------
eps = 0.1
VARIANT = 24
VARIABLES = 40
COUNT = 10

a = -1
b = 1
A = B = 0
    
    
# subs must be a sympy object and result is also the same object
def func_for_partition_1(yk_m1, yk, yk_p1, h):
    func = sympy.sin(VARIANT) * (yk_p1 - 2 * yk + yk_m1) / h ** 2
    func += (1 + sympy.cos(VARIANT) * x ** 2) * yk + 1
    return func

def func_for_partition_2(yk_m1, yk, yk_p1, h):
    func = (yk_p1 - 2 * yk + yk_m1) / h ** 2
    func += (sympy.cos(x) * ((yk_p1 - yk_m1) / (2 * h)))
    func += (5 * (1 + (sympy.sin(x)) ** 2) * yk - 4 * sympy.exp(0.5 * x))
    return func

def func_for_partition_3(yk_m1, yk, yk_p1, h):
    func = (yk_p1 - 2 * yk + yk_m1) / h ** 2
    func -= (yk * x)
    func -= (2 * x)
    return func

def func_for_partition_4(yk_m1, yk, yk_p1, h, k, q):
    func = - k * (yk_p1 - 2 * yk + yk_m1) / h ** 2
    func += (q * yk)
    func -= (7 * (x + 1 / (x + 0.5)))
    return func

# this is the case when we don't have proper value for limit points
# they were calculated by me with my hands
def b_equation_for_partition_3(yn, yn_m1, yn_m2, h):
    func = 3 * yn - 4 * yn_m1 + yn_m2 - 6 * h
    return func

# this is the case when we don't have proper value for limit points
def a_equation_for_partition_3(y0, y1, y2, h):
    func = y0 * (3 + h) - 4 * y1 + y2 - 4.5 * h
    return func

def b_equation_for_partition_4(yn, yn_m1, yn_m2, h):
    func = 0.4 * (3 * yn - 4 * yn_m1 + yn_m2) / (2 * h) + 0.5 * yn;
    return func

def a_equation_for_partition_4(y0, y1, y2, h):
    func = -1.2 * (-y2 + 4 * y1 - 3 * y0) / (2 * h) + 0.5 * y0;
    return func

# subs must be a sympy object and result is also the same object
def func_for_substantiation_1(subs):
    func = sympy.sin(VARIANT) * sympy.diff(subs, x, x)
    func += ((1 + sympy.cos(VARIANT) * x ** 2) * subs + 1)
    return func

def func_for_substantiation_2(subs):
    func = sympy.diff(subs, x, x) + sympy.cos(x) * sympy.diff(subs, x)
    func += (5 * (1 + (sympy.sin(x)) ** 2) * subs - 4 * sympy.exp(0.5 * x))
    return func

def func_for_substantiation_3(subs):
    func = sympy.diff(subs, x, x) - x * subs - 2 * x
    return func

def func_for_substantiation_4(subs, k, q):
    func = - k * sympy.diff(subs, x, x) + q * subs - 7 * (x + 1 / (x + 0.5))
    return func

def generate_basis_sequence(n):
    sequence = []
    for i in range(n):
        # generate the sympy sequence
        sequence.append((x ** i) * (1 - x ** 2))

    return sequence

def generate_basis_sequence_2(n):
    sequence = []
    for i in range(n):
        # generate the sympy sequence
        sequence.append((x ** (2 * i - 2)) * (1 - x ** 2))

    return sequence

# -----------------------------------------------------
# c is upper, b - middle, a - lower, d - solution
def solve_system(c_list, b_list, a_list, d_list):
    n = len(b_list)
    y = [b_list[0]]
    P = [-c_list[0] / b_list[0]]
    Q = [d_list[0] / b_list[0]] 
    
    for i in range(1, n - 1):
        y.append(a_list[i - 1] * P[i - 1] + b_list[i])
        P.append(-c_list[i] / y[i])
        Q.append((d_list[i] - a_list[i - 1] * Q[i - 1])/ y[i])
     
    y.append(a_list[-1] * P[-1] + b_list[-1])
    Q.append((d_list[-1] - a_list[-1] * Q[-1])/ y[-1])
    #  x = [(-a_list[-1] * b_sub[-1] + d_list[-1]) / (b_list[-1] + a_list[-1] * a[-1])]
    x = [0] * n
    x[-1] = Q[-1]
    for i in range(n - 1, 0, -1):
        # this (len(d) - 1) - i  -1 is the trick for x index to count it as 0+, 
        # not 4-
        # x.append(a[i] * x[(n - 1) - i - 1] + b_sub[i])
        x[i - 1] = P[i - 1] * x[i] + Q[i - 1]
    
    x = list(reversed(x))
    return x
# we are getting function with appropriate constants
def build_function_from_basis(basis):
    result = 0
    for i in range(len(basis)):
        current_a = sympy.Symbol('a' + str(i))
        result += current_a * basis[i]
    
    return result
    
def collocations_method(basis, points, func_for_substantiation):
    # unite basis functions to one
    func = build_function_from_basis(basis)
    # create from base func our psi func
    psi_func = func_for_substantiation(func)
    # generate variables for linsolve
    symbols = [sympy.Symbol('a' + str(i)) for i in range(len(points))]
    lin_system = []
    
    # substitude variables for linear system and simplify to linear with evalf
    for point in points:
        lin_system.append(psi_func.subs(x, point).evalf())
    
    # solving our system
    answer = list(list(linsolve(lin_system, *symbols))[0])
    return answer
    
# check for current precision. Note that func_in_points has to be sympy object
def check_convergence(points, func_in_points, true_function, eps):
    for i in range(len(points)):
        # uncomment to watch the real difference
        # print(abs(true_function.subs(x, points[i]) - func_in_points[i]))
        
        if abs(true_function.subs(x, points[i]) - func_in_points[i]) > eps:
            return False
        
    return True

# c is upper, b - middle, a - lower, d - solution
def _solve_func(c_list, b_list, a_list, d_list):
    n = len(b_list)
    y = [b_list[0]]
    P = [-c_list[0] / b_list[0]]
    Q = [d_list[0] / b_list[0]] 
    
    for i in range(1, n - 1):
        y.append(a_list[i - 1] * P[i - 1] + b_list[i])
        P.append(-c_list[i] / y[i])
        Q.append((d_list[i] - a_list[i - 1] * Q[i - 1])/ y[i])
     
    y.append(a_list[-1] * P[-1] + b_list[-1])
    Q.append((d_list[-1] - a_list[-1] * Q[-1])/ y[-1])
    x = [0] * n
    x[-1] = Q[-1]
    for i in range(n - 1, 0, -1):
        # this (len(d) - 1) - i  -1 is the trick for x index to count it as 0+, 
        # not 4-
        x[i - 1] = P[i - 1] * x[i] + Q[i - 1]
    
    return x

def solve_diag_system(lin_system):
    c, b, a, d = [], [], [], []
    cs = sympy.Poly(lin_system[0]).coeffs()
    d.append(-cs[2])
    c.append(cs[1])
    b.append(cs[0])
    
    for i in range(1, len(lin_system) - 1):
        cs = sympy.Poly(lin_system[i]).coeffs()
        d.append(-cs[3])
        c.append(cs[2])
        b.append(cs[1])
        a.append(cs[0])    
    
    cs = sympy.Poly(lin_system[-1]).coeffs()
    d.append(-cs[2])
    b.append(cs[1])
    a.append(cs[0])
    return _solve_func(c, b, a, d)

# check for current precision. Note that func_in_points has to be sympy object
def check_convergence_points(prev_points, current_points, eps, h_rate=2):
    for i in range(len(prev_points)):       
        if abs(current_points[h_rate * i] - prev_points[i]) > eps:
            # uncomment to watch the real difference
            print(abs(current_points[h_rate * i] - prev_points[i]))
            return False
        
    return True

def differences_method(start_variables_count, 
                       a,
                       b, 
                       y_a, 
                       y_b, 
                       func_for_partition, 
                       eps=0.1,
                       verbose=True):
    if verbose:
        print('CALCULATIONS [BEGIN:]')
        
    iteration_count = 0
    
    prev_answer = []
    old_points = []
    while True:
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
        for i in range(1, start_variables_count - 1):
            lin_system.append(func_for_partition(symbols[i - 1], 
                                                 symbols[i], 
                                                 symbols[i + 1],
                                                 h).evalf())
        for i in range(start_variables_count - 2):
            lin_system[i] = lin_system[i].subs(x, points[i])
        
        lin_system[0] = lin_system[0].subs(symbols[0], y_a)
        lin_system[-1] = lin_system[-1].subs(symbols[-1], y_b)
        
        
        
        del symbols[0], symbols[-1]
        
        # solving our system and converting FiniteSet to simple list
        answer = solve_diag_system(lin_system)
        
        # showing the verbose difference between steps
        if iteration_count != 0 and verbose:
            print('ITERATION [%d]' % iteration_count)
        
            D, y = points, answer
            plt.figure()
            plt.plot(D, y, color='red', label='current')
            
            D, y = old_points, prev_answer
            plt.plot(D, y, label='prev')
            
            # this is the ticks for better visualisation
            plt.yticks(linspace(float(min(answer) - abs(min(answer) / 3)), 
                                float(max(answer) + abs(max(answer))), 11))
            plt.title('Iteration_difference: ')
            plt.grid(True)
            plt.show()
        
        # if there is no function to be compated with or we are already get our
        # precision we will break our loop
        if iteration_count != 0 and check_convergence_points(prev_answer, 
                                                             answer,
                                                             eps):
            break
            
        iteration_count += 1
        prev_answer = answer
        old_points = points
        
        # this is done for border points
        start_variables_count -= 2
        start_variables_count *= 2
        
    points.insert(0, a)
    points.append(b)
    
    answer.insert(0, y_a)
    answer.append(y_b)
    
    data_type = namedtuple('data', 
                           ('points', 'answer', 'step', 'iterations_count'))
    
    if verbose:
        print('CALCULATIONS [END:]')
        
    return data_type(points, answer, (b - a) / start_variables_count, 
                     iteration_count)

# this task cannot be solved exactly. So we may try to get function from 
# previous methods as 'True' function and calculate precision according to it
# TASK 1
basis = generate_basis_sequence(COUNT)
k = collocations_method(basis, linspace(a + 0.1, b - 0.1, COUNT), 
                        func_for_substantiation_1)
true_function = 0

for i in range(len(k)):
    true_function += k[i] * basis[i]

data = differences_method(VARIABLES, a, b, A, B, func_for_partition_1, eps)
D, y = data.points, data.answer

# function that was found by collocations method using basis from 1 labwork 
# (this basis is incorrect)
func_y = []

for i in range(len(D)):
    func_y.append(true_function.subs(x, D[i]))
    
plt.figure()
plt.plot(D, y, color='red')
plt.plot(D, func_y, color='red')
plt.yticks(linspace(-1, 1, 10))
plt.title('differences method task_1: ')
plt.grid(True)
plt.show()
print(data.iterations_count, data.step)

# TASK 2
eps = 0.15

a = 0
b = 2
A = 0
B = 4

basis = generate_basis_sequence(COUNT)
k = collocations_method(basis, linspace(a + 0.01, b - 0.01, COUNT), 
                        func_for_substantiation_2)
true_function = 0

for i in range(len(k)):
    true_function += k[i] * basis[i]

data = differences_method(VARIABLES, a, b, A, B, func_for_partition_2, eps)

D, y = data.points, data.answer

# function that was found by collocations method using basis from 1 labwork 
# (this basis is incorrect)
func_y = []
for i in range(len(D)):
    func_y.append(true_function.subs(x, D[i]))
    
plt.figure()
plt.plot(D, y, color='red')
plt.plot(D, func_y, color='blue')
plt.yticks(linspace(-5, 5, 11))
plt.title('differences method task_2: ')
plt.grid(True)
plt.show()
print(data.iterations_count, data.step)

# TASK 3
def differences_method_modifyed(start_variables_count, 
                                a,
                                b,
                                func_for_partition, 
                                first_eq,
                                last_eq,
                                eps=0.01,
                                verbose=True):
    if verbose:
        print('CALCULATIONS [BEGIN:]')
        
    iteration_count = 0
    
    prev_answer = []
    old_points = []
    while True:
        # generate variables for linsolve
        start_variables_count += 2
        symbols = [sympy.Symbol('y' + str(i)) for i in
                   range(start_variables_count)]
        # define our step
        h = (b - a) / start_variables_count 
        # generate x points:
        points = linspace(a + h, b - h, start_variables_count - 2).tolist()
        # build system of equations
        lin_system = [0]
        for i in range(1, start_variables_count - 1):
            lin_system.append(func_for_partition(symbols[i - 1], 
                                                 symbols[i], 
                                                 symbols[i + 1],
                                                 h).evalf())
        
        for i in range(1, start_variables_count - 1):
            lin_system[i] = lin_system[i].subs(x, points[i - 1])
        
        lin_system[0] = first_eq(symbols[0], symbols[1], symbols[2], h)
        lin_system.append(last_eq(symbols[-1], symbols[-2], symbols[-3], h))
        
        # solving our system and converting FiniteSet to simple list
        answer = list(list(linsolve(lin_system, *symbols))[0])
        
        # showing the verbose difference between steps
        if iteration_count != 0 and verbose:
            print('ITERATION [%d]' % iteration_count)
        
            points.insert(0, a)
            points.append(b)
            old_points.insert(0, a)
            old_points.append(b)
        
            D, y = points, answer
            plt.figure()
            plt.plot(D, y, color='red', label='current')
            
            D, y = old_points, prev_answer
            plt.plot(D, y, label='prev')
            
            # this is the ticks for better visualisation
            plt.yticks(linspace(float(min(answer) - abs(min(answer) / 3)), 
                                float(max(answer) + abs(max(answer))), 11))
            plt.title('Iteration_difference: ')
            plt.grid(True)
            plt.show()
            
            del points[0], points[-1], old_points[0], old_points[-1]

        if iteration_count != 0:
            a_a, a_b, p_a, p_b = answer[0], answer[-1], prev_answer[0],\
            prev_answer[-1]
            del answer[0], answer[-1], prev_answer[0], prev_answer[-1]
        # if there is no function to be compated with or we are already get our
        # precision we will break our loop
        if iteration_count != 0 and check_convergence_points(prev_answer, 
                                                             answer,
                                                             eps):
            answer.insert(0, a_a)
            prev_answer.insert(0, p_a)
            answer.append(a_b)
            prev_answer.append(p_b)
            break
        
        if iteration_count != 0:
            answer.insert(0, a_a)
            prev_answer.insert(0, p_a)
            answer.append(a_b)
            prev_answer.append(p_b)
            
        iteration_count += 1
        prev_answer = answer
        old_points = points
        
        # this is done for border points
        start_variables_count -= 2
        start_variables_count *= 2
        
    points.insert(0, a)
    points.append(b)
    
    data_type = namedtuple('data', 
                           ('points', 'answer', 'step', 'iterations_count'))
    
    if verbose:
        print('CALCULATIONS [END:]')
        
    return data_type(points, answer, (b - a) / start_variables_count, 
                     iteration_count)
    
eps = 0.03

a = 1.5
b = 3.5
A = 0
B = 4

basis = generate_basis_sequence(COUNT)
k = collocations_method(basis, linspace(a + 0.01, b - 0.01, COUNT), 
                        func_for_substantiation_3)
true_function = 0

for i in range(len(k)):
    true_function += k[i] * basis[i]

data = differences_method_modifyed(VARIABLES, a, b, func_for_partition_3, 
                                   a_equation_for_partition_3, 
                                   b_equation_for_partition_3,
                                   eps)

D, y = data.points, data.answer
# function that was found by collocations method using basis from 1 labwork 
# (this basis is incorrect)
func_y = []
for i in range(len(D)):
    func_y.append(true_function.subs(x, D[i]))
    
plt.figure()
plt.plot(D, y, color='red')
plt.plot(D, func_y, color='blue')
plt.yticks(linspace(-5, 5, 11))
plt.title('differences method task_3: ')
plt.grid(True)
plt.show()
print(data.iterations_count, data.step)

# TASK 4
def collocations_method_modifyed_points(basis, 
                                        points, 
                                        func_for_substantiation,
                                        c,
                                        k,
                                        q):
    # unite basis functions to one
    func = build_function_from_basis(basis)
    # create from base func our psi func
    psi_func_1 = func_for_substantiation(func, k[0], q[0])
    psi_func_2 = func_for_substantiation(func, k[1], q[1])
    # generate variables for linsolve
    symbols = [sympy.Symbol('a' + str(i)) for i in range(len(points))]
    lin_system = []
    
    # substitude variables for linear system and simplify to linear with evalf
    for point in points:
        if point < c:
            psi_func = psi_func_1
        else:
            psi_func = psi_func_2
            
        lin_system.append(psi_func.subs(x, point).evalf())
    
    # solving our system
    answer = list(list(linsolve(lin_system, *symbols))[0])
    return answer

def differences_method_modifyed_points(start_variables_count, 
                                       a,
                                       b,
                                       c,
                                       k,
                                       q,
                                       func_for_partition, 
                                       first_eq,
                                       last_eq,
                                       eps=0.1,
                                       verbose=True):
    if verbose:
        print('CALCULATIONS [BEGIN:]')
        
    iteration_count = 0
    
    prev_answer = []
    old_points = []
    while True:
        # generate variables for linsolve
        start_variables_count += 2
        symbols = [sympy.Symbol('y' + str(i)) for i in
                   range(start_variables_count)]
        # define our step
        h = (b - a) / start_variables_count 
        # generate x points:
        points = linspace(a + h, b - h, start_variables_count - 2).tolist()
        # build system of equations
        lin_system = [0]
        for i in range(1, start_variables_count - 1):
            if a + i * h < c:
                k_, q_ = k[0], q[0]
            else:
                k_, q_ = k[1], q[1]
                
            lin_system.append(func_for_partition(symbols[i - 1], 
                                                 symbols[i], 
                                                 symbols[i + 1],
                                                 h, k_, q_).evalf())
        
        for i in range(1, start_variables_count - 1):
            lin_system[i] = lin_system[i].subs(x, points[i - 1])
        
        lin_system[0] = first_eq(symbols[0], symbols[1], symbols[2], h)
        lin_system.append(last_eq(symbols[-1], symbols[-2], symbols[-3], h))
        
        # solving our system and converting FiniteSet to simple list
        answer = list(list(linsolve(lin_system, *symbols))[0])
        
        # showing the verbose difference between steps
        if iteration_count != 0 and verbose:
            print('ITERATION [%d]' % iteration_count)
        
            points.insert(0, a)
            points.append(b)
            old_points.insert(0, a)
            old_points.append(b)
        
            D, y = points, answer
            plt.figure()
            plt.plot(D, y, color='red', label='current')
            
            D, y = old_points, prev_answer
            plt.plot(D, y, label='prev')
            
            # this is the ticks for better visualisation
            plt.yticks(linspace(float(min(answer) - abs(min(answer) / 3)), 
                                float(max(answer) + abs(max(answer))), 11))
            plt.title('Iteration_difference: ')
            plt.grid(True)
            plt.show()
            
            del points[0], points[-1], old_points[0], old_points[-1]

        if iteration_count != 0:
            a_a, a_b, p_a, p_b = answer[0], answer[-1], prev_answer[0],\
            prev_answer[-1]
            del answer[0], answer[-1], prev_answer[0], prev_answer[-1]
        # if there is no function to be compated with or we are already get our
        # precision we will break our loop
        if iteration_count != 0 and check_convergence_points(prev_answer, 
                                                             answer,
                                                             eps):
            answer.insert(0, a_a)
            prev_answer.insert(0, p_a)
            answer.append(a_b)
            prev_answer.append(p_b)
            break
        
        if iteration_count != 0:
            answer.insert(0, a_a)
            prev_answer.insert(0, p_a)
            answer.append(a_b)
            prev_answer.append(p_b)
            
        iteration_count += 1
        prev_answer = answer
        old_points = points
        
        # this is done for border points
        start_variables_count -= 2
        start_variables_count *= 2
        
    points.insert(0, a)
    points.append(b)
    
    data_type = namedtuple('data', 
                           ('points', 'answer', 'step', 'iterations_count'))
    
    if verbose:
        print('CALCULATIONS [END:]')
        
    return data_type(points, answer, (b - a) / start_variables_count, 
                     iteration_count)

eps = 0.01
a = 0
b = 1.5
c = 0.925
k = [1.2, 0.4]
q = [8.3, 12]
A = 0
B = 4

basis = generate_basis_sequence(COUNT)
k = collocations_method_modifyed_points(basis, 
                                        linspace(a + 0.01, b - 0.01, COUNT), 
                                        func_for_substantiation_4, c, k, q)
true_function = 0

for i in range(len(k)):
    true_function += k[i] * basis[i]

data = differences_method_modifyed_points(VARIABLES, a, b, c, k, q,
                                          func_for_partition_4, 
                                          a_equation_for_partition_4, 
                                          b_equation_for_partition_4, 
                                          eps)

D, y = data.points, data.answer
# function that was found by collocations method using basis from 1 labwork 
# (this basis is incorrect)
func_y = []
for i in range(len(D)):
    func_y.append(true_function.subs(x, D[i]))
    
plt.figure()
plt.plot(D, y, color='red')
# plt.plot(D, func_y, color='blue')
plt.yticks(linspace(-0.5, 0.5, 7))
plt.title('differences method task_4: ')
plt.grid(True)
plt.show()
