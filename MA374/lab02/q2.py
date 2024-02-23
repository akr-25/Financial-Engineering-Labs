# Author: Aman Kumar-200123007
# Question 2:

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

def calcCallPrice(M, r, S0, sigma, T, K, uFunc, dFunc):
  delta_t = T/M

  # Calculate u, r, d and p*

  R = np.exp(r * delta_t)
  u = uFunc(sigma, r, delta_t)
  d = dFunc(sigma, r, delta_t)

  rnp = (R - d)/(u - d) # Risk neutral probability of up move
  
  # Calculate all the max value, end value and probability of the path
  def rec(step, maxi, curr):
    if(step == M):
        return max(0, (maxi - K))
    else:
        return (rec(step + 1, max(maxi , curr*u), curr*u)*rnp + rec(step + 1, max(maxi, curr*d), curr*d)*(1-rnp))/R

  return rec(0, S0, S0)

def calcPutPrice(M, r, S0, sigma, T, K, uFunc, dFunc):
  delta_t = T/M

  # Calculate u, r, d and p*

  R = np.exp(r * delta_t)
  u = uFunc(sigma, r, delta_t)
  d = dFunc(sigma, r, delta_t)

  rnp = (R - d)/(u - d) # Risk neutral probability of up move
  
  # Calculate all the max value, end value and probability of the path
  def rec(step, mini, curr):
    if(step == M):
        return max(0, (K - mini))
    else:
        return (rec(step + 1, min(mini , curr*u), curr*u)*rnp + rec(step + 1, min(mini, curr*d), curr*d)*(1-rnp))/R

  return rec(0, S0, S0)

def initialize():
    return 100, 100, 1, 10, 0.08, 0.20


S0, K, T, M, r, sigma = initialize()
S0_array = range(S0 - 30, S0 + 30, 1)
call_prices = []
put_prices = []
for S0 in S0_array:
    c = calcCallPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    p = calcPutPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    call_prices.append(c)
    put_prices.append(p)

plt.plot(S0_array, call_prices, label="Call Prices")
plt.plot(S0_array, put_prices, label="Put Prices")
plt.xlabel("S0")
plt.ylabel("Option Prices")
plt.title(f"This is for second type of u d functions")
plt.legend()
plt.savefig(f"q2/S0.png")
plt.clf()
# plt.show()

S0, K, T, M, r, sigma = initialize()
K_array = range(K - 30, K + 30, 1)
call_prices = []
put_prices = []
for K in K_array:
    c = calcCallPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    p = calcPutPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    call_prices.append(c)
    put_prices.append(p)

plt.plot(K_array, call_prices, label="Call Prices")
plt.plot(K_array, put_prices, label="Put Prices")
plt.xlabel("K")
plt.ylabel("Option Prices")
plt.title(f"This is for second type of u d functions")
plt.legend()
plt.savefig(f"q2/K.png")
plt.clf()
# plt.show()

S0, K, T, M, r, sigma = initialize()
r_array = np.linspace(0.1, 1, 100)
call_prices = []
put_prices = []
for r in r_array:
    c = calcCallPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    p = calcPutPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    call_prices.append(c)
    put_prices.append(p)

plt.plot(r_array, call_prices, label="Call Prices")
plt.plot(r_array, put_prices, label="Put Prices")
plt.xlabel("r")
plt.ylabel("Option Prices")
plt.title(f"This is for second type of u d functions")
plt.legend()
plt.savefig(f"q2/r.png")
plt.clf()
# plt.show()

S0, K, T, M, r, sigma = initialize()
sigma_array = np.linspace(0.1, 1, 100)
call_prices = []
put_prices = []
for sigma in sigma_array:
    c = calcCallPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    p = calcPutPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
    call_prices.append(c)
    put_prices.append(p)

plt.plot(sigma_array, call_prices, label="Call Prices")
plt.plot(sigma_array, put_prices, label="Put Prices")
plt.xlabel("sigma")
plt.ylabel("Option Prices")
plt.title(f"This is for second type of u d functions")
plt.legend()
plt.savefig(f"q2/sigma.png")
plt.clf()
# plt.show()

S0, K, T, M, r, sigma = initialize()
for K in [95, 100, 105]:
    M_array = range(5, 12)
    call_prices = []
    put_prices = []
    for M in M_array:
        c = calcCallPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
        p = calcPutPrice(M, r, S0, sigma, T, K, uCalc1, dCalc1)
        call_prices.append(c)
        put_prices.append(p)

    plt.plot(M_array, call_prices, label="Call Prices")
    plt.xlabel("M")
    plt.ylabel("Option Prices")
    plt.title(f"This is for second type of u d functions, K = {K}")
    plt.legend()
    plt.savefig(f"q2/M_K{K}_call.png")
    plt.clf()
    # plt.show()

    plt.plot(M_array, put_prices, label="Put Prices")
    plt.xlabel("M")
    plt.ylabel("Option Prices")
    plt.title(f"This is for second type of u d functions, K = {K}")
    plt.legend()
    plt.savefig(f"q2/M_K{K}_put.png")
    plt.clf()
    # plt.show()


