import numpy as np
import matplotlib.pyplot as plt

def f(x): # the function you want to analyze
    return np.exp(np.sin(x))

def df(x): # the function you want to analyze
    return np.exp(np.sin(x)) * np.cos(x)

def d2f(x): # Analytical derivative
    return np.exp(np.sin(x)) * (np.cos(x)**2 - np.sin(x))

def finiteDiff(a,b,n): 
    # uses second order centered difference to approximate the derivative
    # inputs: a - start, b - stop, n - number of nodes 
    # Outputs: x - mesh, df_approx - approximation of first derivative
    x = np.linspace(a, b, n, endpoint = False)
    h = (b-a)/(n-1)
    y = f(x)
    df_approx = np.zeros(y.size)
    for i in range(y.size):
        df_approx[i] = (y[(i+1)%y.size] - 2*y[i] + y[i-1]) / h**2 # The modulo (%) here lets us enforce periodicity on the array (python will loop back when over indexed)
    return x, df_approx

def rel_error(a, b, n):
    x, d2f_approx = finiteDiff(a,b,n)
    d2f_exact = d2f(x) # get the analytical solution
    L_2_error = np.linalg.norm(d2f_exact-d2f_approx)/np.linalg.norm(d2f_exact)
    L_inf_error = np.linalg.norm(d2f_exact-d2f_approx,ord=np.inf)/np.linalg.norm(d2f_exact,ord=np.inf)
    return L_2_error, L_inf_error

def main():
    # Part 1
    a = 0
    b = 2*np.pi
    x_exact = np.linspace(a, b, 1000, endpoint = False)
    d2f_exact = d2f(x_exact) # Calculate the analytical solution
    x_approx, d2f_approx = finiteDiff(0, 2*np.pi, 50) # approximate using FD + 21 nodes

    plt.rcParams['lines.linewidth'] = 2.5 # set a wider line width for all plots in this script
    ax = plt.axes()
    plt.figure(1)
    plt.plot(x_exact, d2f_exact, label = "Exact Solution")
    plt.plot(x_approx, d2f_approx, 'r--', linewidth = 6, label = "N=21 FD-Approx")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(a, b)
    plt.legend()
    plt.savefig('Images\\problem2_1.png')

    # Part 2
    num_trials = 16
    meshes = np.floor(np.logspace(1, 4, num_trials)).astype(int)
    L_2 = np.zeros(num_trials)
    L_inf = np.zeros(num_trials)
    for i in range(num_trials):
        L_2[i], L_inf[i] = rel_error(0, 2*np.pi, meshes[i]) #calculate relative errors for increasingly finer meshes.

    plt.figure(2)
    ax = plt.axes()
    ax.set_xlabel("Number of FD Nodes")
    ax.set_ylabel("Relative Error")
    plt.loglog(meshes, L_2, label = "L_2")
#    plt.loglog(meshes, L_inf, 'r--', linewidth = 6, label = "L_inf")
    plt.legend()
    plt.savefig('Images\\problem2_2.png')
    return 0

if __name__ == "__main__":
    main()

