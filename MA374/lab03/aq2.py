import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt
import time

def lookback_price(i, S0, u, d, M):
    path = format(i, 'b').zfill(M) 
    cmax = S0
    s = S0 
    for j in path:
        s *= int(j)*d + (1 - int(j))*u
        cmax = max(cmax, s)
    return cmax-s

def loopback_option(S0, T, M, r, sigma):
    print(f"Case M = {M}")
    start = time.time()
    
    t = T/M
    u = exp(sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = exp(-sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)  
    R = exp(r*t)

    p = (R - d)/(u - d);

    if d >= exp(r*t) or exp(r*t) >= u:
        print(f"Arbitrage exists for M       \t= {M}")
        return 0, 0

    V = []
    # initialize array 
    for i in range(0, M + 1):
        V.append(np.zeros(int(np.power(2,i))))
    
    # initialize final values
    for i in range(int(np.power(2, M))):
        req_price = lookback_price(i, S0, u, d, M)
        V[M][i] = max(req_price, 0)
    
    # fill intermediate values 
    for j in range(M - 1, -1, -1):
        for i in range(0, int(np.power(2, j))):
            V[j][i] = ( p*V[j + 1][2*i] + (1-p)*V[j + 1][2*i + 1]) / R;

 
    print(f"Price of Loopback Option at t = 0          \t= {V[0][0]}")
    end = time.time()
    print(f"Execution time of basic binomial algorithm \t= {end - start} seconds\n")

        
    return V[0][0]


def print_loopback_option(S0, T, M, r, sigma):
    t = T/M
    u = exp(sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = exp(-sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)  
    R = exp(r*t)

    p = (R - d)/(u - d);

    if d >= exp(r*t) or exp(r*t) >= u:
        print(f"Arbitrage exists for M       \t= {M}")
        return 0, 0

    V = []
    # initialize array 
    for i in range(0, M + 1):
        V.append(np.zeros(int(np.power(2,i))))
    
    # initialize final values
    for i in range(int(np.power(2, M))):
        req_price = lookback_price(i, S0, u, d, M)
        V[M][i] = max(req_price, 0)
    
    # fill intermediate values 
    for j in range(M - 1, -1, -1):
        for i in range(0, int(np.power(2, j))):
            V[j][i] = ( p*V[j + 1][2*i] + (1-p)*V[j + 1][2*i + 1]) / R;

    for i in range(len(V)):
        print(f"At t = {i}  --->")
        for j in range(len(V[i])):
            print(f"Index no = {j}\tPrice = {V[i][j]}")
    print()



print("For part A ------> ")
M = [5, 10]
prices = []
for m in M:
    prices.append(loopback_option(100, 1, m, 0.08, 0.20))

print("\n\nRunning For part B ..... ")

plt.plot(M, prices)
plt.xlabel("M")
plt.ylabel("Option prices at t = 0") 
plt.title("Option Prices t = 0 v/s M")
plt.show()

prices.clear()
M = [i for i in range(1,16)]
for m in range(1, 16):
    prices.append(loopback_option(100, 1, m, 0.08, 0.20))

plt.plot(M, prices)
plt.xlabel("M")
plt.ylabel("Option prices at t = 0") 
plt.title("Option Prices t = 0 v/s M")
plt.show()

print("\n\nFor part C ------>")
print_loopback_option(100, 1, 5, 0.08, 0.20)
