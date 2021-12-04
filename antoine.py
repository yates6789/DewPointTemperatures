from re import I
from CoolProp.CoolProp import PropsSI, get_global_param_string
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
import pandas as pd


def array_pressure(fluid, n):
    return np.linspace(min_pressure(fluid), crit_pressure(fluid), n)

def crit_pressure(fluid):
    return PropsSI("Pcrit", fluid)

def min_pressure(fluid):
    return PropsSI('PMIN', fluid)

def sat_temp(P, fluid):
    return PropsSI('T', 'Q', 1, 'P', P, fluid)

def condition_press_and_temp(fluid, n):
    P = array_pressure(fluid, n)
    T = sat_temp(P, fluid)
    P = P[T != None]
    T = T[T != None]
    P = P[abs(T) != np.inf]
    T = T[abs(T) != np.inf]
    return P, T

def r_squared(Y, Y_fit):
    correlation_matrix = np.corrcoef(Y, Y_fit)
    correlation_xy = correlation_matrix[0,1]
    return correlation_xy**2


def func_antoine(P, A, B, C):
    return np.divide(A, np.log(P) + B) + C

def string_antoine(constants):
    return f'{round(constants[0], 2)}/(log(P) + {round(constants[1], 2)}) + {round(constants[2], 2)}'

def eval_antoine(P, constants):
    return func_antoine(P, constants[0], constants[1], constants[2])

def fit_antoine(P, T, initial_guess):
    popt, pcov = curve_fit(func_antoine, xdata=P, ydata=T, p0=initial_guess, maxfev=1_000)
    return popt, r_squared(T, eval_antoine(P, popt))

def random_walk_algorithm(P, T):
    r_squared = 0
    while r_squared < 0.999:
        initial_guess = [random.randint(-1000, 1000), random.randint(-1000, 1000), random.randint(-1000, 1000)]
        popt, r_squared = fit_antoine(P, T, initial_guess)
    return popt, r_squared
        

def plot_results(P, T, T_antoine, antoine_r_squared, fluid):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coolprop, = ax.plot(P/100_000, T)
    coolprop.set_label('Coolprop')
    antoine_plot, = ax.plot(P/100_000, T_antoine, '--')
    antoine_plot.set_label(f'T = A/(log(P) + B) + C')
    ax.legend()
    plt.grid()
    plt.title(f"{fluid} - Dew Point Temperature - R2 = {str(antoine_r_squared)[0:4]}")
    plt.xlabel('Sat. Pressure - [bar]')
    plt.ylabel('Sat. Temperature - [K]')
    plt.savefig(f'{fluid} - Dewpoint Curve')
    


fluids = get_global_param_string("FluidsList").split(',')
print(f'\n\n {fluids} \n\n')
for fluid in fluids:
    print(f'\n\n {fluid} \n\n')
    Import = pd.read_csv('antoine_results.csv', header=None).to_numpy()
    existing_fluids = Import[:, 0]
    if fluid in existing_fluids:
        continue
    pressure, temperature = condition_press_and_temp(fluid, 5000)
    antoine, antoine_r_squared = random_walk_algorithm(pressure, temperature)  
    plot_results(pressure, temperature, eval_antoine(pressure, antoine), antoine_r_squared, fluid) 
    new_row = np.array([fluid, antoine[0], antoine[1], antoine[2], pressure[0], pressure[-1], temperature[0], temperature[-1], antoine_r_squared])
    Export = np.vstack((Import, new_row))
    pd.DataFrame(Export).to_csv('antoine_results.csv', index=None, header=None)





