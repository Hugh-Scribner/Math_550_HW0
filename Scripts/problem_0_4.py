import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def u(x,y): # The exact solution
    return np.sin(y)*np.cos(x)

def f(x,y): # The forcing function
    return -2*np.sin(y)*np.cos(x)

def finiteDifferencePoisson(a, b, n, bc="dirichlet"):
    if bc == "Neumann":
        print("The script isn't ready to handle Neumann Boundary conditions yet")
        return 0
    else:
        A = np.eye(n**2)
        x = np.linspace(a, b, n)
        y = np.linspace(a, b, n) # this line in unnecessary but makes my code a bit easier to read
        xx, yy = np.meshgrid(x,y)
        h = (b-a)/(n-1)
        b_hat_inter = f(xx,yy)
        b_hat = b_hat_inter.reshape((n**2,))
        print(b_hat_inter)
        print(b_hat)
        print(xx)
        print(yy)
        print("Defining linear system...")
        for i in range(n):
            for j in range(n):
                k = n*i+j
                # leave A unmodified (we want an identity row) and update b_hat to enforce the boundary condition along the boundaries
                if i == 0:
                    b_hat[k] = np.sin(y[j])
                elif j == 0:
                    b_hat[k] = 0
                elif i == n-1:
                    b_hat[k] = np.sin(y[j])*np.cos(5)
                elif j == n-1 :
                    b_hat[k] = np.sin(5)*np.cos(x[i])
                # Finite difference formula everywhere else
                else:
                    A[k,k] = -4/h**2
                    A[k,k-1] = 1/h**2
                    A[k,k+1] = 1/h**2
                    A[k,k-n] = 1/h**2
                    A[k,k+n] = 1/h**2
    print("Linear system defined")
    # Solve the system
    print("Solving linear system...")
    u_approx = np.linalg.solve(A, b_hat)
    print("Linear system solved")
    # Reshape outputs so they are good for plotting

    u_approx = u_approx.reshape((n,n))
    return xx, yy, u_approx
        
def main():
    a = 0
    b = 5
    n = 40

    # Get the numerical solution
    xx,yy,u_approx = finiteDifferencePoisson(a, b, n)

    # Get the exact solution
    u_exact = u(xx,yy)

    fig1 = plt.figure(1)
    ax1 = plt.axes(projection='3d')
    ax1.plot_surface(xx,yy,u_approx)
    plt.savefig("Images\\problem4_1.png")

    fig2 = plt.figure(2)
    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(xx,yy,u_exact)
    plt.savefig("Images\\problem4_2.png")
    plt.show()

if __name__ == "__main__":
    main()