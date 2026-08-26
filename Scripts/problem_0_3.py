import numpy as np
import matplotlib.pyplot as plt

def main():
    # Part 1
    a = 0 # Left endpoint
    b = 5 # Right endpoint
    num_nodes = 5
    h = (b-a)/(num_nodes-1)
    x = np.linspace(a, b, num_nodes)
    print(x)

    # Build the "mesh"
    A = ((1/h**2 - np.sin(x)/(2*h)) * np.eye(num_nodes, k=-1) + 
            (1 - 2/(h**2)) * np.eye(num_nodes) + 
            (1/h**2 + np.sin(x)/(2*h)) * np.eye(num_nodes, k=1))
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
    print(A)
    print(b_hat)

    plt.figure(1)
    plt.plot(x, u_approx,'r--', label = "FD-Approx")
    plt.plot(x, u_exact, label = "Exact")
    plt.legend()
    plt.savefig("Images\\problem3_1.png")

    return 0

if __name__ == "__main__":
    main()