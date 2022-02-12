from math import sqrt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas
from random import gauss
import numpy as np


#README
# This is a simple implement of Vasicek model calibration.
# Data to fit was provided from this page:
# https://fred.stlouisfed.org/categories/33003
# To run the code:
# 1) Provide the path to your datafile and the column name with values in the main function.
# 2) Adjust the upper bound.
# 3) Run the code.
#
# Possible outcomes:
# 1) Plot,
# 2) curve_fit runtime error -> rerun with newly generated parameters,
# 3) curve_fit value error -> choose a smaller upper bound.


#data import and cleanup
def LIBOR_data_import(path_to_file, values_column_title):
    df = pandas.read_csv(path_to_file)
    X = list(df[values_column_title])
    Y = [float(element) for element in X if element != '.']
    return np.array(Y)


#ito sums for stochastic integral in Vasicek solution
def Ito_sums(t, W, sample_size):
    ito = []
    ito.append(W[0])
    for i in range(1,sample_size):
        ito.append(np.exp(t[i-1])*(W[i]-W[i-1])+ito[i-1])
    return ito


#implementation of EM scheme to Vasicek equation
def EM_Vasicek(delta, wiener, a, b, sigma, initial_value, sample_size):
    X = [0 for i in range(sample_size)]
    X[0] = initial_value
    for i in range(1, sample_size):
        X[i]=X[i-1]+a*(b-X[i-1])*delta+sigma*(wiener[i]-wiener[i-1])

    return X


#implementation of the solution to Vasicek equation with ito
def Vasicek_solution_function_approximated(t_parameters, a, b, sigma):
    t, Wt, r0, ito = t_parameters
    return r0*np.exp(-a*t)+b*(1-np.exp(-a*t))+sigma*np.exp(-(2*a*t))*ito


#brownian motion generator
def generate_wiener(length, delta):
    W = [0]
    for i in range(1,length):
        W.append(W[i-1]+ sqrt(delta)*gauss(0, 1))
    
    return np.array(W)


def main():
    #initial data
    upper_bound = 1
    X = LIBOR_data_import('libor.csv', 'USD3MTD156N')
    sample_size = len(X)
    delta = upper_bound/sample_size

    #parameters
    t = np.linspace(0, upper_bound, sample_size)
    Wt = generate_wiener(sample_size, delta)
    r0 = np.array([X[0] for i in range(sample_size)])
    ito = np.array(Ito_sums(t, Wt, sample_size))
    
    #calibration by nonlinear least squares method
    a, b, sigma  = curve_fit(Vasicek_solution_function_approximated, (t, Wt, r0, ito), X)[0]
    
    #test
    fitted_X = EM_Vasicek(delta, Wt, a, b, sigma, X[0], sample_size)
    plt.plot(t, X, color='r', label='initial_data')
    plt.plot(t, fitted_X, color='b', label='generated_data')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
