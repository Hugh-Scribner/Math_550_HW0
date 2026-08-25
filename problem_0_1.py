import numpy as np
import matplotlib.pyplot as plt

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

def main():
    # Part 1
    x_exact = np.linspace(0, 2*np.pi, 1001)
    dy_exact = df(x_exact) # Calculate the analytical solution
    x_approx, df_approx = finiteDiff12(0,2*np.pi, 21) # approximate using FD + 21 nodes

    plt.rcParams['lines.linewidth'] = 2 # set a wider line width for all plots in this script
    plt.figure(1)
    plt.plot(x_exact, dy_exact, label = "Exact Solution")
    plt.plot(x_approx, df_approx, 'r--', label = "N=21 FD-Approx")
    plt.legend()
    plt.savefig('Images\problem1_1.png')

    # Part 2
    num_trials = 16
    meshes = np.floor(np.logspace(1, 6, num_trials)).astype(int)
    L_2 = np.zeros(num_trials)
    L_inf = np.zeros(num_trials)
    for i in range(num_trials):
        L_2[i], L_inf[i] = rel_error(0, 2*np.pi, meshes[i])

    plt.figure(2)
    plt.loglog(meshes, L_2, label = "L_2")
    plt.loglog(meshes, L_inf, label = "L_inf")
    plt.legend()
    plt.savefig('Images\problem1_2.png')

    return 0

if __name__ == "__main__":
    main()

