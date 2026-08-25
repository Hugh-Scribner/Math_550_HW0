import numpy as np
import matplotlib.pyplot as plt

def f(x): # the function you want to analyze
    return np.exp(np.sin(x))

def df(x): # Analytical derivative
    return np.exp(np.sin(x)) * np.cos(x)

def finiteDiff12(a,b,n): 
    # uses second order centered difference to approximate the derivative
    # inputs: a - start, b - stop, n - number of nodes 
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
    return x, df_approx, y

def main():
    x_exact = np.linspace(0, 2*np.pi, 1001)
    dy_exact = df(x_exact)
    x_approx, df_approx, y = finiteDiff12(0,2*np.pi, 201)

    plt.plot(x_approx, df_approx)
    plt.plot(x_exact, dy_exact)
    plt.show()

    return 0

if __name__ == "__main__":
    main()

