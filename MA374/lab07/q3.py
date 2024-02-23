import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D


def BSM(S, K, T, r, t, sigma, option='both', q=0):
    if np.max(t) > T:
        raise ValueError('t must be less than T')

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    callPrice = S * np.exp(-q * (T-t)) * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    putPrice = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S * np.exp(-q * (T-t)) * norm.cdf(-d1)

    if option == 'call':
        return callPrice
    elif option == 'put':
        return putPrice
    else:
        return callPrice, putPrice


def C(t, s):
    return BSM(s, 1, 1, 0.05, t, 0.6, 'call')

def P(t, s):
    return BSM(s, 1, 1, 0.05, t, 0.6, 'put')

def plot_3D():
    s = np.linspace(0.1, 1.9, 100)
    t = np.linspace(0, 1, 100)
    S, T = np.meshgrid(s, t)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(S, T, C(T, S), label='C(t, s)')
    ax.plot_surface(S, T, P(T, S), label='P(t, s)')
    ax.set_xlabel('s')
    ax.set_ylabel('t')
    ax.set_zlabel('C(t, s) and P(t, s)')
    ax.set_title('C(t, s) and P(t, s) as a function of both t and s')
    plt.savefig('image/q3/3d.png')
    plt.close()

if __name__ == '__main__':
    plot_3D()