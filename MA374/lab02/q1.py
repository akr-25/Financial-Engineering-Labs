# Author: Aman Kumar-200123007
# Question 1:

# Imports
import numpy as np
from matplotlib import pyplot as plt

def uCalc1(sigma, r, delta_t):
    return np.exp(sigma * np.sqrt(delta_t) + (r - 0.5 * np.power(sigma, 2)) * (delta_t))

def uCalc2(sigma, r, delta_t):
    return np.exp(sigma * np.sqrt(delta_t))

def dCalc1(sigma, r, delta_t):
    return np.exp(-sigma * np.sqrt(delta_t) + (r - 0.5 * np.power(sigma, 2)) * (delta_t))

def dCalc2(sigma, r, delta_t):
    return np.exp(-sigma * np.sqrt(delta_t))

def calcOptionPrice(M, r, S0, sigma, T, K, uFunc, dFunc):
  delta_t = T/M

  # Calculate u, r, d and p*

  R = np.exp(r * delta_t)
  u = uFunc(sigma, r, delta_t)
  d = dFunc(sigma, r, delta_t)

  rnp = (R - d)/(u - d) # Risk neutral probability of up move

  # Initialize the call option price at maturity
  C = np.zeros((M+1, M+1))
  for j in range(M+1):
    C[j, M] = max(0, S0 * np.power(u, j) * np.power(d, M-j) - K)

  # Calculate the call option price using backward induction
  for i in range(M-1, -1, -1):
    for j in range(i+1):
      C[j, i] = (rnp * C[j+1, i+1] + (1-rnp) * C[j, i+1])/R

  # Initialize the put option price at maturity
  P = np.zeros((M+1, M+1))
  for j in range(M+1):
    P[j, M] = max(0, K - S0 * np.power(u, j) * np.power(d, M-j))

  # Calculate the put option price using backward induction
  for i in range(M-1, -1, -1):
    for j in range(i+1):
      P[j, i] = (rnp * P[j+1, i+1] + (1-rnp) * P[j, i+1])/R

  
  return C[0, 0], P[0, 0]

def initialize():
    return 100, 100, 1, 100, 0.08, 0.20

uFuncs = [uCalc2, uCalc1]
dFuncs = [dCalc2, dCalc1]
sss = ["first", "second"]

for funcIdx in range(2):  
    S0, K, T, M, r, sigma = initialize()
    S0_array = range(S0 - 30, S0 + 30, 1)
    call_prices = []
    put_prices = []
    for S0 in S0_array:
        c, p = calcOptionPrice(M, r, S0, sigma, T, K, uFuncs[funcIdx], dFuncs[funcIdx])
        call_prices.append(c)
        put_prices.append(p)

    plt.plot(S0_array, call_prices, label="Call Prices")
    plt.plot(S0_array, put_prices, label="Put Prices")
    plt.xlabel("S0")
    plt.ylabel("Option Prices")
    plt.title(f"This is for {sss[funcIdx]} type of u d functions")
    plt.legend()
    plt.savefig(f"q1/{sss[funcIdx]}/S0.png")
    plt.clf()
    # plt.show()

    S0, K, T, M, r, sigma = initialize()
    K_array = range(K - 30, K + 30, 1)
    call_prices = []
    put_prices = []
    for K in K_array:
        c, p = calcOptionPrice(M, r, S0, sigma, T, K, uFuncs[funcIdx], dFuncs[funcIdx])
        call_prices.append(c)
        put_prices.append(p)

    plt.plot(K_array, call_prices, label="Call Prices")
    plt.plot(K_array, put_prices, label="Put Prices")
    plt.xlabel("K")
    plt.ylabel("Option Prices")
    plt.title(f"This is for {sss[funcIdx]} type of u d functions")
    plt.legend()
    plt.savefig(f"q1/{sss[funcIdx]}/K.png")
    plt.clf()
    # plt.show()

    S0, K, T, M, r, sigma = initialize()
    r_array = np.linspace(0.1, 1, 100)
    call_prices = []
    put_prices = []
    for r in r_array:
        c, p = calcOptionPrice(M, r, S0, sigma, T, K, uFuncs[funcIdx], dFuncs[funcIdx])
        call_prices.append(c)
        put_prices.append(p)

    plt.plot(r_array, call_prices, label="Call Prices")
    plt.plot(r_array, put_prices, label="Put Prices")
    plt.xlabel("r")
    plt.ylabel("Option Prices")
    plt.title(f"This is for {sss[funcIdx]} type of u d functions")
    plt.legend()
    plt.savefig(f"q1/{sss[funcIdx]}/r.png")
    plt.clf()
    # plt.show()

    S0, K, T, M, r, sigma = initialize()
    sigma_array = np.linspace(0.1, 1, 100)
    call_prices = []
    put_prices = []
    for sigma in sigma_array:
        c, p = calcOptionPrice(M, r, S0, sigma, T, K, uFuncs[funcIdx], dFuncs[funcIdx])
        call_prices.append(c)
        put_prices.append(p)

    plt.plot(sigma_array, call_prices, label="Call Prices")
    plt.plot(sigma_array, put_prices, label="Put Prices")
    plt.xlabel("sigma")
    plt.ylabel("Option Prices")
    plt.title(f"This is for {sss[funcIdx]} type of u d functions")
    plt.legend()
    plt.savefig(f"q1/{sss[funcIdx]}/sigma.png")
    plt.clf()
    # plt.show()

    S0, K, T, M, r, sigma = initialize()
    for K in [95, 100, 105]:
        M_array = range(50, 150)
        call_prices = []
        put_prices = []
        for M in M_array:
            c, p = calcOptionPrice(M, r, S0, sigma, T, K, uFuncs[funcIdx], dFuncs[funcIdx])
            call_prices.append(c)
            put_prices.append(p)

        plt.plot(M_array, call_prices, label="Call Prices")
        plt.xlabel("M")
        plt.ylabel("Option Prices")
        plt.title(f"This is for {sss[funcIdx]} type of u d functions, K = {K}")
        plt.legend()
        plt.savefig(f"q1/{sss[funcIdx]}/M_K{K}_call.png")
        plt.clf()
        plt.plot(M_array, put_prices, label="Put Prices")
        plt.xlabel("M")
        plt.ylabel("Option Prices")
        plt.title(f"This is for {sss[funcIdx]} type of u d functions, K = {K}")
        plt.legend()
        plt.savefig(f"q1/{sss[funcIdx]}/M_K{K}_put.png")
        plt.clf()
        # plt.show()


