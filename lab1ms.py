#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sympy
from sympy.solvers.solveset import linsolve

from numpy import linspace
import functools


x = sympy.Symbol('x')


VARIANT = 24
VARIABLES = 10
DIFF_POINTS = 5

a = -1
b = 1


# subs must be a sympy object and result is also the same object
def func_for_substantiation(subs):
    func = sympy.sin(VARIANT) * sympy.diff(subs, x, x) + ((1 + \
                    sympy.cos(VARIANT) * x ** 2) * subs + 1)
    return func

def generate_basis_sequence(n):
    sequence = []
    for i in range(n):
        # generate the sympy sequence
        sequence.append((x ** i) * (1 - x ** 2))

    return sequence
    
# we are getting function with appropriate constants
def build_function_from_basis(basis):
    result = 0
    for i in range(len(basis)):
        current_a = sympy.Symbol('a' + str(i))
        result += current_a * basis[i]
    
    return result
    
def collocations_method(basis, points):
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
    answer = linsolve(lin_system, *symbols)
    return answer

def integral_least_square_method(basis, a, b):
    # unite basis functions to one
    func = build_function_from_basis(basis)
    # create from base func our psi func
    psi_func = func_for_substantiation(func)
    # generate variables for linsolve
    symbols = [sympy.Symbol('a' + str(i)) for i in range(len(basis))]
    lin_system = []
    
    # getting our integral system and simplify equations to linear with evalf
    for i in range(len(basis)):
        lin_system.append(sympy.integrate(2 * sympy.diff(
                psi_func, symbols[i]) * psi_func, (x, a, b)).evalf())
    
    # solving our system
    answer = linsolve(lin_system, *symbols)
    return answer

def discrete_least_square_method(basis, points_num, a, b):
    # unite basis functions to one
    func = build_function_from_basis(basis)
    # create from base func our psi func
    psi_func = func_for_substantiation(func)
    # let's build our sum 
    seq = [psi_func.subs(x, point) ** 2 for point 
           in linspace(a + 0.05, b - 0.05, points_num)]
    
    psi_sqr_sum = functools.reduce((lambda a, b: a + b), seq)
    # generate variables for linsolve
    symbols = [sympy.Symbol('a' + str(i)) for i in range(len(basis))]
    lin_system = []
    
    # getting our integral system and simplify equations to linear with evalf
    for i in range(len(basis)):
        lin_system.append(sympy.diff(psi_sqr_sum, symbols[i]).evalf())
    
    # solving our system
    answer = linsolve(lin_system, *symbols)
    return answer

def galerkin_method(basis, a, b):
    # unite basis functions to one
    func = build_function_from_basis(basis)
    # create from base func our psi func
    psi_func = func_for_substantiation(func)
    # generate variables for linsolve
    symbols = [sympy.Symbol('a' + str(i)) for i in range(len(basis))]
    lin_system = []
    
    # getting our integral system and simplify equations to linear with evalf
    for i in range(len(basis)):
        lin_system.append(sympy.integrate(
                psi_func * basis[i], (x, a, b)).evalf())
    
    # solving our system
    answer = linsolve(lin_system, *symbols)
    return answer

print(linspace(a + 0.2, b - 0.2, VARIABLES))

print('collocations method: \n',
      collocations_method(generate_basis_sequence(VARIABLES), 
                          linspace(a + 0.2, b - 0.2, VARIABLES)))

print('integral lsm method: \n',
      integral_least_square_method(generate_basis_sequence(VARIABLES), a, b))

print('discrete lsm method: \n',
      discrete_least_square_method(generate_basis_sequence(VARIABLES),
                                   VARIABLES + DIFF_POINTS, a, b))

print('galerkin method: \n',
      galerkin_method(generate_basis_sequence(VARIABLES), a, b))
