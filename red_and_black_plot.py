import matplotlib.pyplot as plt
import numpy as np


def phi_function(x, p):
    if x == 1 or x == 0:
        return x
    elif x < 0.5:
        return p*phi_function(2*x, p)
    else:
        return p+(1-p)*phi_function(2*x-1, p)
     

def plotting():
    p = 0.5
    x = np.linspace(0, 1, num=100)
    y = np.array([phi_function(argument, p) for argument in x])
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    plotting()
