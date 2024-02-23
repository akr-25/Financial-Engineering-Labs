# Author: Aman Kumar-200123007 
# Question 3:

# Imports
import numpy as np
import tabulate as tb

# Given variables 
S0 = 100
r = 0.05
sigma = 0.3
T = 5
K = 105

M = 20
delta_t = T/M

# Calculate u, r, d and p*

R = np.exp(r * delta_t)
u = np.exp(sigma * np.sqrt(delta_t) + (r - 0.5 * np.power(sigma, 2)) * (delta_t))
d = np.exp(-sigma * np.sqrt(delta_t) + (r - 0.5 * np.power(sigma, 2)) * (delta_t))

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


# Tabulate the option prices at different times
time = [0, 0.5, 1, 1.5, 3, 4.5]
for t in time:
  put_price = []
  call_price = []
  S_no = []
  print (f'Option prices at time {t}:')
  for i in range(M+1):
    if int(t/delta_t) + 1 == i:
      break
    call_price.append(C[i, int(t/delta_t)])
    put_price.append(P[i, int(t/delta_t)])
    S_no.append(i + 1)
  print(tb.tabulate({'S.no': S_no, 'Call Price': call_price, 'Put Price': put_price}, headers='keys', tablefmt='psql'))

