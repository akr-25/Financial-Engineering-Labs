import numpy as np
import matplotlib.pyplot as plt


def simulate_geometric_brownian_motion(initial_price, drift, volatility, num_periods):
    dt = 1.0 / num_periods
    returns = np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.normal(size=num_periods))
    prices = initial_price * np.cumprod(returns)
    return prices



def variance_reduction(option_payoff, control_variate, r, n, dt):
    # Calculate mean of control variate and option payoff
    X_bar = np.mean(control_variate)
    Y_bar = np.mean(option_payoff)

    # Calculate numerator and denominator for regression coefficient
    num = np.sum((control_variate - X_bar) * (option_payoff - Y_bar))
    denom = np.sum((control_variate - X_bar) ** 2)

    # Calculate regression coefficient
    b = num / denom

    # Calculate reduced variate
    reduced_variate = option_payoff - b * (control_variate - X_bar) * np.exp(-r * n * dt)

    return reduced_variate


def asian_option_price(S_0, r, sigma, K, max_iter=1000, path_length=126, n=126):
    """
    Calculate prices and variances of Asian call and put options using Monte Carlo simulation and variance reduction.

    Args:
    S_0 (float): initial asset price
    r (float): risk-free interest rate
    sigma (float): asset volatility
    K (float): option strike price
    max_iter (int): number of simulations to run (default: 1000)
    path_length (int): length of each simulated stock price path (default: 126)
    n (int): number of averaging periods (default: 126)

    Returns:
    tuple: (call_price, put_price, call_var, put_var), where:
        call_price (float): price of Asian call option
        put_price (float): price of Asian put option
        call_var (float): variance of Asian call option price estimate
        put_var (float): variance of Asian put option price estimate
    """

    dt = 1 / 252

    # Initialize lists to store option payoffs and control variates
    call_option_payoff = []
    put_option_payoff = []
    control_variate_call = []
    control_variate_put = []

    # Run Monte Carlo simulations
    for i in range(max_iter):
        # Simulate stock price path
        S = simulate_geometric_brownian_motion(S_0, r, sigma, path_length)

        # Calculate option payoffs
        avg_S = np.mean(S)
        V_call = max(avg_S - K, 0)
        V_put = max(K - avg_S, 0)

        # Add discounted option payoff to lists
        call_option_payoff.append(np.exp(-r * n * dt) * V_call)
        put_option_payoff.append(np.exp(-r * n * dt) * V_put)

        # Calculate control variates and add to lists
        control_variate_call.append(np.exp(-r * n * dt) * max(K - S[-1], 0))
        control_variate_put.append(np.exp(-r * n * dt) * max(S[-1] - K, 0))

    # Apply variance reduction to option payoffs
    call_option_payoff = variance_reduction(call_option_payoff, control_variate_call, r, n, dt)
    put_option_payoff = variance_reduction(put_option_payoff, control_variate_put, r, n, dt)

    # Calculate prices and variances of option payoffs
    call_price = np.mean(call_option_payoff)
    put_price = np.mean(put_option_payoff)
    call_var = np.var(call_option_payoff)
    put_var = np.var(put_option_payoff)

    return call_price, put_price, call_var, put_var



def variation_with_S0(r, sigma, K):
  S0 = np.linspace(50, 150, num=250)
  call, put = np.array([]), np.array([])

  for i in S0:
    call_price, put_price, _, _ = asian_option_price(i, r, sigma, K, 500, 150, 100)
    call = np.append(call, call_price)
    put = np.append(put, put_price)
  
  plt.plot(S0, call)
  plt.xlabel("Initial asset price (S0)")
  plt.ylabel("Asian call option price")
  plt.title("Dependence of Asian Call Option on S0")
  plt.show()

  plt.plot(S0, put)
  plt.xlabel("Initial asset price (S0)")
  plt.ylabel("Asian put option price")
  plt.title("Dependence of Asian Put Option on S0")
  plt.show()

  return call, put


def variation_with_K(S0, r, sigma):
  K = np.linspace(50, 150, num=250)
  call, put = np.array([]), np.array([])

  for i in K:
    call_price, put_price, _, _ = asian_option_price(S0, r, sigma, i, 500, 150, 100)
    call = np.append(call, call_price)
    put = np.append(put, put_price)
  
  plt.plot(K, call)
  plt.xlabel("Strike price (K)")
  plt.ylabel("Asian call option price")
  plt.title("Dependence of Asian Call Option on K")
  plt.show()

  plt.plot(K, put)
  plt.xlabel("Strike price (K)")
  plt.ylabel("Asian put option price")
  plt.title("Dependence of Asian Put Option on K")
  plt.show()

  return call, put


def variation_with_r(S0, sigma, K):
  r = np.linspace(0, 0.5, num=120, endpoint=False)
  call, put = np.array([]), np.array([])


  for i in r:
    call_price, put_price, _, _ = asian_option_price(S0, i, sigma, K, 500, 150, 100)
    call = np.append(call, call_price)
    put = np.append(put, put_price)


  plt.plot(r, call)
  plt.xlabel("Risk-free rate (r)")
  plt.ylabel("Asian call option price")
  plt.title("Dependence of Asian Call Option on r")
  plt.show()

  plt.plot(r, put)
  plt.xlabel("Risk-free rate (r)")
  plt.ylabel("Asian put option price")
  plt.title("Dependence of Asian Put Option on r")
  plt.show()

  return call, put


def variation_with_sigma(S0, r, K):
  sigma = np.linspace(0, 1, num=120, endpoint=False)
  call, put = [], []

  for i in sigma:
    call_price, put_price, _, _ = asian_option_price(S0, r, i, K, 500, 150, 100)
    call.append(call_price)
    put.append(put_price)
  
  plt.plot(sigma, call)
  plt.xlabel("Volatility (sigma)")
  plt.ylabel("Asian call option price")
  plt.title("Dependence of Asian Call Option on sigma")
  plt.show()

  plt.plot(sigma, put)
  plt.xlabel("Volatility (sigma)")
  plt.ylabel("Asian put option price")
  plt.title("Dependence of Asian Put Option on sigma")
  plt.show()

  return call, put

for K in [90, 105, 110]:
  call_price, put_price, call_var, put_var = asian_option_price(100, 0.05, 0.2, K)
  print("\n\n************** For K = {} **************".format(K))
  print("Asian call option price \t\t=", call_price)
  print("Variance in Asian call option price \t=", call_var)
  print()
  print("Asian put option price \t\t\t=", put_price)
  print("Variance in Asian put option price \t=", put_var)

# Sensitivity Analysis
variation_with_S0(0.05, 0.2, 105)
variation_with_K(100, 0.05, 0.2)
variation_with_r(100, 0.2, 105)
variation_with_sigma(100, 0.05, 105)