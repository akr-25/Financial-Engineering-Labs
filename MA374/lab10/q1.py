import numpy as np
import matplotlib.pyplot as plt

def simulate_stock_path(initial_price, drift, volatility, title):
    num_periods = 252
    time_grid = np.arange(num_periods)

    for _ in range(10):
        stock_prices = simulate_geometric_brownian_motion(initial_price, drift, volatility, num_periods)
        plt.plot(time_grid, stock_prices)

    plt.xlabel('Time (days)')
    plt.ylabel('Stock Price')
    plt.title(title)
    plt.show()


def simulate_geometric_brownian_motion(initial_price, drift, volatility, num_periods):
    dt = 1.0 / num_periods
    returns = np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * np.random.normal(size=num_periods))
    prices = initial_price * np.cumprod(returns)
    return prices

def asian_option_price(initial_price, risk_free_rate, volatility, strike_price, num_iterations=1000, path_length=126, num_periods=126):
    time_step = 1/252
    call_option_payoffs, put_option_payoffs = [], []

    for i in range(num_iterations):
        stock_prices = simulate_geometric_brownian_motion(initial_price, risk_free_rate, volatility, path_length)
        average_price = np.mean(stock_prices)
        call_option_payoff = max(average_price - strike_price, 0)
        put_option_payoff = max(strike_price - average_price, 0)

        call_option_payoffs.append(np.exp(-risk_free_rate*num_periods*time_step) * call_option_payoff)
        put_option_payoffs.append(np.exp(-risk_free_rate*num_periods*time_step) * put_option_payoff)

    call_option_price = np.mean(call_option_payoffs)
    put_option_price = np.mean(put_option_payoffs)
    call_option_variance = np.var(call_option_payoffs)
    put_option_variance = np.var(put_option_payoffs)

    return call_option_price, put_option_price, call_option_variance, put_option_variance

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


# Simulate stock paths
stock_path_labels = ["Asset price in real world", "Asset price in risk-neutral world"]
stock_path_parameters = [(100, 0.1, 0.2), (100, 0.05, 0.2)]
for i in range(len(stock_path_labels)):
    label = stock_path_labels[i]
    parameters = stock_path_parameters[i]
    simulate_stock_path(*parameters, title=label)

# Calculate Asian option prices
strikes = [90, 105, 110]
for K in strikes:
    call_price, put_price, call_var, put_var = asian_option_price(100, 0.05, 0.2, K)
    print("\n\n************** For K = {} **************".format(K))
    print("Asian call option price \t\t=", call_price)
    print("Variance in Asian call option price \t=", call_var)
    print()
    print("Asian put option price \t\t\t=", put_price)
    print("Variance in Asian put option price \t=", put_var)

# Sensitivity analysis
variation_with_S0(0.05, 0.2, 105)
variation_with_K(100, 0.05, 0.2)
variation_with_r(100, 0.2, 105)
variation_with_sigma(100, 0.05, 105)
