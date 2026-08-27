import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def u(x,y): # The exact solution
    return np.sin(y)*np.cos(x)

def f(x,y): # The forcing function
    return -2*np.sin(y)*np.cos(x)

def finiteDifferencePoisson(a, b, n, bc="dirichlet", returnGrid = True):
    if bc == "neumann":
        # This method uses forward + backward differences to enforce the neumann conditions... this will mean the solution to the PDE will be O(h)
        # The Neumann boundary solver is not intended to be used wtih large matricies.
        A = sp.lil_array((n**2,n**2))
        h = (b-a)/(n-1)
        x = np.linspace(a-h, b+h, n)
        y = np.linspace(a-h, b+h, n) # this line in unnecessary but makes my code a bit easier to read
        xx, yy = np.meshgrid(x,y)
        b_hat_inter = f(xx,yy)
        b_hat = b_hat_inter.reshape((n**2,))
        for i in range(n):
            for j in range(n):
                k = n*i+j
                # leave A unmodified (we want an identity row) and update b_hat to enforce the boundary condition along the boundaries
                if i == 0: # u_y(x,0)=0, forward difference
                    A[k,k] = -1/h
                    A[k,k+n] = 1/h
                    b_hat[k] = 0
                elif j == 0: # u_x(0,y)=0, forward difference
                    A[k,k] = -1/h
                    A[k,k+1] = 1/h
                    b_hat[k] = 0
                elif i == n-1: # u_y(x,5)=cos5*cosx, backward difference
                    A[k,k-n] = -1/h
                    A[k,k] = 1/h
                    b_hat[k] = np.cos(5)*np.cos(x[j])
                elif j == n-1 : #u_x(5,y) = cos5*cosx, backward difference
                    A[k,k-1] = -1/h
                    A[k,k] = 1/h
                    b_hat[k] = np.cos(y[i])*np.cos(5)
                # Finite difference formula everywhere else
                else:
                    A[k,k-n] = 1/h**2
                    A[k,k-1] = 1/h**2
                    A[k,k] = -4/h**2
                    A[k,k+1] = 1/h**2
                    A[k,k+n] = 1/h**2
        print(f"the condition number of A is {np.linalg.cond(A.toarray())}")
    else:
        A = sp.lil_array((n**2,n**2))
        h = (b-a)/(n-1)
        x = np.linspace(a, b, n)
        y = np.linspace(a, b, n) # this line in unnecessary but makes my code a bit easier to read
        xx, yy = np.meshgrid(x,y)
        b_hat_inter = f(xx,yy)
        b_hat = b_hat_inter.reshape((n**2,))
        print("Defining Linear System...")
        for i in range(n):
            for j in range(n):
                k = n*i+j
                # leave A unmodified (we want an identity row) and update b_hat to enforce the boundary condition along the boundaries
                if i == 0: # u(x,0)
                    A[k,k] = 1
                    b_hat[k] = 0
                elif j == 0: # u(0,y)
                    A[k,k] = 1
                    b_hat[k] = np.sin(y[i])
                elif i == n-1: # u(x,5)
                    A[k,k] = 1
                    b_hat[k] = np.sin(5)*np.cos(x[j])
                elif j == n-1 : #u(5,y)
                    A[k,k] = 1
                    b_hat[k] = np.sin(y[i])*np.cos(5)
                # Finite difference formula everywhere else
                else:
                    A[k,k-n] = 1/h**2
                    A[k,k-1] = 1/h**2
                    A[k,k]   = -4/h**2
                    A[k,k+1] = 1/h**2
                    A[k,k+n] = 1/h**2

    # Solve the system
    print("Linear System Defined, Solving...")
    u_approx = sp.linalg.spsolve(A.tocsr(), b_hat)
    print("Solution Found")
    # Reshape outputs so they are good for plotting
    if returnGrid:
        u_approx = u_approx.reshape((n,n))
    return xx, yy, u_approx

def rel_error(a, b, n):
    xx, yy, f_approx = finiteDifferencePoisson(a,b,n, returnGrid = False)
    f_exact = f(xx, yy).reshape((n**2,)) # get the analytical solution
    L_2_error = np.linalg.norm(f_exact-f_approx)/np.linalg.norm(f_exact)
    L_inf_error = np.linalg.norm(f_exact-f_approx,ord=np.inf)/np.linalg.norm(f_exact,ord=np.inf)
    return L_2_error, L_inf_error
        
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
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Numerical Solution")
    plt.savefig("Images\\problem4_1.png")

    fig2 = plt.figure(2)
    ax2 = plt.axes(projection='3d')
    ax2.plot_surface(xx,yy,u_exact)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Exact Solution")
    plt.savefig("Images\\problem4_2.png")

    num_trials = 5
    meshes = np.floor(np.logspace(1, 5, num_trials)).astype(int)
    L_2 = np.zeros(num_trials)
    L_inf = np.zeros(num_trials)
    for i in range(num_trials):
        print(f"Simulating mesh with {meshes[i]**2} nodes")
        L_2[i], L_inf[i] = rel_error(a, b, meshes[i]) #calculate relative errors for increasingly finer meshes.

    plt.figure(3)
    plt.loglog(meshes, L_2, label = "L_2")
    plt.loglog(meshes, L_inf, label = "L_inf")
    plt.legend()
    plt.savefig('Images\\problem4_3.png')

    xx,yy,u_neumann = finiteDifferencePoisson(a, b, n, bc="neumann") # the condition number of A is 1.174812990360801e+17 for one n=40

    fig4 = plt.figure(4)
    ax4 = plt.axes(projection='3d')
    ax4.plot_surface(xx,yy,u_neumann)
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("Neummann Numerical Solution")
    plt.savefig("Images\\problem4_4.png")


if __name__ == "__main__":
    main()