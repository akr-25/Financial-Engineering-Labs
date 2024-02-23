# Author: Aman Kumar-200123007
# Question 2:

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Given variables 
S0 = 100
r = 0.05
sigma = 0.3
T = 5
K = 105

for jump in [1, 5]:
  results = {'M': [], 'Call Price': [], 'Put Price': []}
  if jump == 1:
    end = 201
  else:
    end = 501
  for M in range(1, end, jump):
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

    
    results['M'].append(M)
    results['Call Price'].append(C[0, 0])
    results['Put Price'].append(P[0, 0])

  # Plot the results using matplotlib
  plt.plot(results['M'], results['Call Price'], label='Call Price')
  plt.plot(results['M'], results['Put Price'], label='Put Price')
  plt.xlabel('M')
  plt.ylabel('Price')
  plt.legend()
  plt.show()
