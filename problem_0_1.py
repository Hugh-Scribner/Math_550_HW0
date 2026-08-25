import numpy as np

def f(x):
    y = np.exp(np.sin(x))
    return y

def df(x):
    y = np.exp(np.sin(x)) * np.cos(x)
    return y

def main():
    return 0

if __name__ == "__main__":
    main()

