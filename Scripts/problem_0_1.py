import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from fractions import Fraction

def f(x): # the function you want to analyze
    return np.exp(np.sin(x))

def df(x): # Analytical derivative
    return np.exp(np.sin(x)) * np.cos(x)

def finiteDiff12(a,b,n): 
    # uses second order centered difference to approximate the derivative
    # inputs: a - start, b - stop, n - number of nodes 
    # Outputs: x - mesh, df_approx - approximation of first derivative
    x = np.linspace(a, b, n)
    y = f(x)
    df_approx = np.zeros(x.size)

    for i in range(x.size):
        if i == 0: # Use a forward difference for left boundary point
            df_approx[i] = (y[i+1]-y[i])/(x[i+1]-x[i])
        elif i == x.size - 1: # Use backward difference for right boundary point
            df_approx[i] = (y[i]-y[i-1])/(x[i]-x[i-1]) 
        else: # Use a centered difference every where else
            df_approx[i] = (y[i+1]-y[i-1])/(x[i+1]-x[i-1])
    return x, df_approx

def rel_error(a, b, n):
    x, df_approx = finiteDiff12(a,b,n)
    df_exact = df(x) # get the analytical solution
    L_2_error = np.linalg.norm(df_exact-df_approx)/np.linalg.norm(df_exact)
    L_inf_error = np.linalg.norm(df_exact-df_approx,ord=np.inf)/np.linalg.norm(df_exact,ord=np.inf)
    return L_2_error, L_inf_error

def pi_formatter(val, pos):
    # Written by Sonnet
    frac = Fraction(val / np.pi).limit_denominator(10)
    if frac.numerator == 0:
        return "0"
    if frac.denominator == 1:
        n = frac.numerator
        return r"$\pi$" if n == 1 else r"$-\pi$" if n == -1 else rf"${n}\pi$"
    num, den = frac.numerator, frac.denominator
    if num == 1:
        return rf"$\pi/{den}$"
    if num == -1:
        return rf"$-\pi/{den}$"
    return rf"${num}\pi/{den}$"

def main():
    # Part 1
    a = 0
    b = 2*np.pi
    x_exact = np.linspace(a, b, 1001)
    df_exact = df(x_exact) # Calculate the analytical solution
    x_approx, df_approx = finiteDiff12(a, b, 21) # approximate using FD + 21 nodes

    plt.rcParams['lines.linewidth'] = 2.5 # set a wider line width for all plots in this script
    ax = plt.axes()
    plt.figure(1)
    plt.plot(x_exact, df_exact, label = "Exact Solution")
    plt.plot(x_approx, df_approx, 'r--', linewidth = 6, label = "N=21 FD-Approx")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.xaxis.set_major_locator(MultipleLocator(np.pi/2))
    ax.xaxis.set_major_formatter(FuncFormatter(pi_formatter))
    ax.set_xlim(a, b)
    plt.legend()
    plt.savefig('Images\\problem1_1.png')

    # Part 2
    num_trials = 16
    meshes = np.floor(np.logspace(1, 4, num_trials)).astype(int)
    L_2 = np.zeros(num_trials)
    L_inf = np.zeros(num_trials)
    for i in range(num_trials):
        L_2[i], L_inf[i] = rel_error(0, 2*np.pi, meshes[i])

    plt.figure(2)
    ax = plt.axes()
    ax.set_xlabel("Number of FD Nodes")
    ax.set_ylabel("Relative Error")
    plt.loglog(meshes, L_2, label = "L_2")
    plt.loglog(meshes, L_inf, 'r--', linewidth = 6, label = "L_inf")
    plt.legend()
    plt.savefig('Images\\problem1_2.png')

    return 0

if __name__ == "__main__":
    main()

