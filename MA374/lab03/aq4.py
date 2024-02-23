import numpy as np
import math
import matplotlib.pyplot as plt
from functools import reduce
import operator as op
import time

def nCr(n, r):
  r = min(r, n-r)
  numer = reduce(op.mul, range(n, n-r, -1), 1)
  denom = reduce(op.mul, range(1, r+1), 1)
  return numer // denom            


def compute_option_price(i, S0, u, d, M):
    path = format(i, 'b').zfill(M)
    for shift in path:
        S0 *= int(shift)*d + (1-int(shift))*u
    return S0


def efficient_bin_model(S0, K, T, M, r, sigma):
    curr_time = time.time()
    t = T/M
    u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)

    R = math.exp(r*t)
    p = (R - d)/(u - d);

    if d >= math.exp(r*t) and math.exp(r*t) >= u:
        print(f"Arbitrage Opportunity exists for M = {M}")
        return 0,0 


    S = [[0 for i in range(M + 1)] for j in range(M + 1)]

    for i in range(0, M + 1):
        S[M][i] = max(0, S0*math.pow(u, M - i)*math.pow(d, i) - K)

    for j in range(M - 1, -1, -1):
        for i in range(0, j + 1):
            S[j][i] = (p*S[j + 1][i] + (1 - p)*S[j + 1][i + 1]) / R;
 
    print(f"European Call Option \t\t= {S[0][0]}")
    print(f"Execution Time \t\t\t= {time.time() - curr_time} sec\n")

        
    return S[0][0]

def print_efficient_bin_model(S0, K, T, M, r, sigma):
    t = T/M
    u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)

    R = math.exp(r*t)
    p = (R - d)/(u - d);

    if d >= math.exp(r*t) and math.exp(r*t) >= u:
        print(f"Arbitrage Opportunity exists for M = {M}")
        return 0,0 

    S = [[0 for i in range(M + 1)] for j in range(M + 1)]

    for i in range(0, M + 1):
        S[M][i] = max(0, S0*math.pow(u, M - i)*math.pow(d, i) - K)

    for j in range(M - 1, -1, -1):
        for i in range(0, j + 1):
            S[j][i] = (p*S[j + 1][i] + (1 - p)*S[j + 1][i + 1]) / R;


    for i in range(M + 1):
        print(f"At t = {i}")
        for j in range(i + 1):
            print(f"Index no = {j}\tPrice = {S[i][j]}")
        print()
        


def bin_model_unoptimised(S0, K, T, M, r, sigma, display):
    print(f"For M = {M} ->\n")
    curr_time = time.time()
    
    u, d = 0, 0
    t = T/M
    u = math.exp(sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)
    d = math.exp(-sigma*math.sqrt(t) + (r - 0.5*sigma*sigma)*t)  

    R = math.exp(r*t)
    p = (R - d)/(u - d);

    if d >= math.exp(r*t) and math.exp(r*t) >= u:
        if display == 1:
            print(f"Arbitrage Opportunity exists for M = {M}")
            return 0, 0
    else:
        if display == 1:
            print(f"No arbitrage exists for M = {M}")

    S = []
    for i in range(0, M + 1):
        S.append(np.zeros(int(pow(2, i))))
        
    for i in range(int(pow(2, M))):
        cost = compute_option_price(i, S0, u, d, M)
        S[M][i] = max(cost - K, 0)
    
    for j in range(M - 1, -1, -1):
        for i in range(0, int(pow(2, j))):
            S[j][i] = (p*S[j + 1][2*i] + (1 - p)*S[j + 1][2*i + 1]) / R;

    if display == 1: 
        print(f"European Call Option \t\t= {S[0][0]}")
        print(f"Execution Time \t\t\t= {time.time() - curr_time} sec\n")
        
    return S[0][0]



# sub-part (a)
print("\n\nFor part A ------>")

M = [5, 10, 25, 50]
temp1, temp2, temp3 = [], [], []

print('Unoptimised Binomial Algorithm executing------>')
for m in M:
    if m != 25 and m != 50: 
        temp1.append(bin_model_unoptimised(100, 1, 100, m, 0.08, 0.20, 1))


print('\n\nEfficient Binomial Algorithm executing (Markov Based)------>')
for m in M:
    temp2.append(efficient_bin_model(S0 = 100, T = 1, K = 100, M = m, r = 0.08, sigma = 0.20))


print("\n\nRunning For part B ..... ")

plt.plot(M, temp2)
plt.xlabel("M")
plt.ylabel("Option prices at t = 0") 
plt.title("Option Prices t = 0 v/s M")
plt.show()

temp1.clear()
M = [i for i in range(1, 31)]
for m in range(1, 31):
    temp1.append(efficient_bin_model(S0 = 100, T = 1, K = 100, M = m, r = 0.08, sigma = 0.20))

plt.plot(M, temp1)
plt.xlabel("M")
plt.ylabel("Option prices at t = 0") 
plt.title("Option Prices t = 0 v/s M")
plt.show()

# sub-part (c)
print("\n\n-----------------------  sub-part(c)  -----------------------")
print_efficient_bin_model(100, 1, 100, 5, 0.08, 0.20)

