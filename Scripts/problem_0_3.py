import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def finiteDiffSolver(a,b,num_nodes):
    # Build the "mesh
    h = (b-a)/(num_nodes-1)
    x = np.linspace(a, b, num_nodes)
    
    # Solce the PDE
    A = np.diag(1/h**2 - np.sin(x[1:])/(2*h), k=-1) + np.eye(num_nodes)*(1 - 2/(h**2)) + np.diag(1/h**2 + np.sin(x[:-1])/(2*h), k=1)
    b_hat = np.sin(x) * np.cos(x)
    
    # Add in the boundary conditions
    A[0,:] = np.concatenate(([1], np.zeros(num_nodes - 1)))
    A[-1,:] = np.concatenate((np.zeros(num_nodes - 1), [1]))
    b_hat[0] = f(a)
    b_hat[-1] = f(b)
    
    # Solve the ODE
    u_approx = np.linalg.solve(A,b_hat)

    return x, u_approx

def rel_error(a, b, n):
    x, f_approx = finiteDiffSolver(a,b,n)
    f_exact = f(x) # get the analytical solution
    L_2_error = np.linalg.norm(f_exact-f_approx)/np.linalg.norm(f_exact)
    L_inf_error = np.linalg.norm(f_exact-f_approx,ord=np.inf)/np.linalg.norm(f_exact,ord=np.inf)
    return L_2_error, L_inf_error

def main():
    # Part 1
    a = 0 # Left endpoint
    b = 5 # Right endpoint
    num_nodes = 51
    x, u_approx = finiteDiffSolver(a,b,num_nodes)

    # solve the exact solution
    u_exact = f(x)

    plt.rcParams['lines.linewidth'] = 3 # set a wider line width for all plots in this script
    plt.figure(1)
    ax = plt.axes()
    ax.set_xlim(a, b)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.plot(x, u_exact, label = "Exact")
    plt.plot(x, u_approx,'r--', label = "FD-Approx", linewidth=6)
    plt.legend()
    plt.savefig("Images\\problem3_1.png")

# Part 2
    num_trials = 4
    meshes = np.floor(np.logspace(1, 4, num_trials)).astype(int)
    L_2 = np.zeros(num_trials)
    L_inf = np.zeros(num_trials)
    for i in range(num_trials):
        L_2[i], L_inf[i] = rel_error(a, b, meshes[i]) #calculate relative errors for increasingly finer meshes.

    plt.figure(2)
    ax = plt.axes()
    ax.set_xlabel("Number of FD Nodes")
    ax.set_ylabel("Relative Error")
    plt.loglog(meshes, L_2, label = "L_2")
    #plt.loglog(meshes, L_inf, label = "L_inf")
    plt.legend()
    plt.savefig('Images\\problem3_2.png')

    return 0

if __name__ == "__main__":
    main()