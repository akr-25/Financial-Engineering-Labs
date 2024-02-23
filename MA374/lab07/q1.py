import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def BSM(S, K, T, r, t, sigma, option, q=0):
    if t > T:
        raise ValueError('t must be less than T')
    
    if t == T:
        if option == 'call':
            return max(S-K, 0)
        elif option == 'put':
            return max(K-S, 0)
        else :
            raise ValueError('option must be call or put')

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * (T-t)) / (sigma * np.sqrt(T-t))
    d2 = d1 - sigma * np.sqrt(T-t)
    if option == 'call':
        price = S * np.exp(-q * (T-t)) * norm.cdf(d1) - K * np.exp(-r * (T-t)) * norm.cdf(d2)
    elif option == 'put':
        price = K * np.exp(-r * (T-t)) * norm.cdf(-d2) - S * np.exp(-q * (T-t)) * norm.cdf(-d1)
    else :
        raise ValueError('option must be call or put')
    return price

