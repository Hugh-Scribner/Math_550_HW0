import numpy as np
import matplotlib.pyplot as plt

def rel_error(a, b, n):
    x, d2f_approx = finiteDiff(a,b,n)
    d2f_exact = d2f(x) # get the analytical solution
    L_2_error = np.linalg.norm(d2f_exact-d2f_approx)/np.linalg.norm(d2f_exact)
    L_inf_error = np.linalg.norm(d2f_exact-d2f_approx,ord=np.inf)/np.linalg.norm(d2f_exact,ord=np.inf)
    return L_2_error, L_inf_error

def main():
    # Part 1
    a = 0 # Left endpoint
    b = 5 # Right endpoint
    num_nodes = 51
    h = (b-a)/(num_nodes-1)
    x = np.linspace(a, b, num_nodes)

    # Build the "mesh"
    A = np.diag(1/h**2 - np.sin(x[1:])/(2*h), k=-1) + np.eye(num_nodes)*(1 - 2/(h**2)) + np.diag(1/h**2 + np.sin(x[:-1])/(2*h), k=1)
    b_hat = np.sin(x) * np.cos(x)

    # Add in the boundary conditions
    A[0,:] = np.concatenate(([1], np.zeros(num_nodes - 1)))
    A[-1,:] = np.concatenate((np.zeros(num_nodes - 1), [1]))
    b_hat[0] = 0
    b_hat[-1] = np.sin(5)

    # Solve the ODE
    u_approx = np.linalg.inv(A) @ b_hat

    # solve the exact solution
    u_exact = np.sin(x)

    plt.rcParams['lines.linewidth'] = 2 # set a wider line width for all plots in this script
    plt.figure(1)
    plt.plot(x, u_exact, label = "Exact")
    plt.plot(x, u_approx,'r--', label = "FD-Approx")
    plt.legend()
    plt.savefig("Images\\problem3_1.png")

    return 0

if __name__ == "__main__":
    main()