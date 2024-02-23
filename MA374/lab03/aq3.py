from math import exp, sqrt
import matplotlib.pyplot as plt
import time


def func(state, u, d, p, R, M, ST, curr_max, V):
  if state == M + 1 or (ST, curr_max) in V[state]:
    return

  func(state + 1, u, d, p, R, M, ST*u, max(ST*u, curr_max), V)
  func(state + 1, u, d, p, R, M, ST*d, max(ST*d, curr_max), V)

  if state == M:
    V[M][(ST, curr_max)] = max(curr_max - ST, 0)
  else:
    V[state][(ST, curr_max)] = (p*V[state + 1][ (u * ST, max(u * ST, curr_max)) ] + (1 - p)*V[state + 1][ (d * ST, curr_max) ]) / R
  

def eff_lookback_option(S0, T, M, r, sigma):
    print(f"Case M = {M} \n")
    start = time.time()

    u, d = 0, 0
    t = T/M
    u = exp(sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = exp(-sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)  
    p = (exp(r*t) - d)/(u - d);


    V = []
    for i in range(0, M + 1):
        V.append(dict())

    func(0, u, d, p, exp(r*t), M, S0, S0, V)
    
    
    if d >= exp(r*t) or exp(r*t) >= u:
        print(f"Arbitrage Opportunity exists for M = {M}")
        return 0, 0

    print(f"Initial Price of Loopback Option \t= {V[0][ (S0, S0) ]}")
    end = time.time()
    print(f"Execution Time \t\t\t\t= {end - start} sec\n")


    return V[0][ (S0, S0) ]

def print_eff_lookback_option(S0, T, M, r, sigma):
    t = T/M
    u = exp(sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = exp(-sigma*sqrt(t) + (r - 0.5*sigma*sigma)*t)  
    p = (exp(r*t) - d)/(u - d);


    V = []
    for i in range(0, M + 1):
        V.append(dict())

    func(0, u, d, p, exp(r*t), M, S0, S0, V)
    
    
    if d >= exp(r*t) or exp(r*t) >= u:
        print(f"Arbitrage Opportunity exists for M = {M}")
        return 0, 0
    
    i = 0 
    while i < len(V):
        print(f"At t = {i}")
        for key, value in V[i].items():
            print(f"Intermediate state = {key}\t\tPrice = {value}")
        print()
        i += 1




print("\n\nFor part A ------>")
M = [5, 10, 25, 50]
prices = []

for m in M:
    prices.append(eff_lookback_option(100, 1, m, 0.08, 0.20))


print("\n\nRunning For part B ..... ")

plt.plot(M, prices)
plt.xlabel("M")
plt.ylabel("Option prices at t = 0") 
plt.title("Option Prices t = 0 v/s M")
plt.show()

prices.clear()
M = [i for i in range(1, 31)]
for m in range(1, 31):
    prices.append(eff_lookback_option(100, 1, m, 0.08, 0.20))

plt.plot(M, prices)
plt.xlabel("M")
plt.ylabel("Option prices at t = 0") 
plt.title("Option Prices t = 0 v/s M")
plt.show()


print("\n\nFor part C ------>")
print_eff_lookback_option(100, 1, 5, 0.08, 0.20)
