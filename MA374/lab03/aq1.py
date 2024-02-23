import numpy as np
import matplotlib.pyplot as plt

def f(t, sigma, r):
    return np.exp(sigma*pow(t, 0.5) + (r - 0.5*pow(sigma, 2))*t),  np.exp(-1*sigma*pow(t, 0.5) + (r - 0.5*pow(sigma, 2))*t)

def put_helper(S_0 = 100, K = 100, M = 100, T = 1, sigma= 0.2, r= 0.08): 
    M = int(M)
    t = T / M 

    U, D = f(t, sigma, r)   
    p = (np.exp(r*t) - D) / (U - D)

    put_V = []
    for i in range(0, M + 1):
        put_V.append(max(K - S_0*pow(U, M - i)*pow(D, i), 0))

    while (len(put_V) != 1):
        n = len(put_V)
        curr_V = []
        for i in range(0, n - 1):
            curr = S_0*pow(U, n - 2 - i)*pow(D, i)
            curr_V.append(max(K -curr , 0))
        
        put_prices = [] 
        for i in range(1, len(put_V)):
            put_prices.append(max((put_V[i - 1]*p + put_V[i]*(1 - p)) / np.exp(r*t), curr_V[i - 1]))

        put_V = put_prices
    
    return put_V[0]

def call_helper(S_0 = 100, K = 100, M = 100, T = 1, sigma= 0.2, r= 0.08):
    M = int(M)
    t = T / M 
    U, D = f(t, sigma, r)   
    p = (np.exp(r*t) - D) / (U - D)

    call_V = []
    for i in range(0, M + 1):
        call_V.append(max(S_0*pow(U, M - i)*pow(D, i) - K, 0))

    while (len(call_V) != 1):
        n = len(call_V)
        curr_V = []
        for i in range(0, n - 1):
            curr = S_0*pow(U, n - 2 - i)*pow(D, i)
            curr_V.append(max(curr - K, 0))

        call_prices = []
        for i in range(1, len(call_V)):
            call_prices.append(max((call_V[i - 1]*p + call_V[i]*(1 - p)) / np.exp(r*t), curr_V[i - 1]))
    

        call_V = call_prices
    
    return call_V[0]

def plot_f(type, param, start, stop, steps = 100, K = 100):
    put_list = []
    call_list = []

    dict = {}
    if (len(param) == 2):
        dict[param[1]] = K
    
    x_axis = np.linspace(start, stop, num = steps)
    for i in x_axis:
        dict[param[0]] = i
        call_price, put_price  = call_helper(**dict), put_helper(**dict)

        call_list.append(call_price)
        put_list.append(put_price)

    if (type == 1):
        plt.suptitle(f"Varying {param[0]}")

    if (type == 2):
        plt.suptitle(f"Varying {param[0]}, K = {K}")

    plt.subplot(1, 2, 1)
    plt.title("For Call Option")
    plt.plot(x_axis, call_list)

    plt.subplot(1, 2, 2)
    plt.title("For Put Option")
    plt.plot(x_axis, put_list)

    plt.show()

print("Call option price for default parameters: ", call_helper(100, 100, 100, 1, 0.2, 0.08))
print("Put option price for default parameters: ", put_helper(100, 100, 100, 1, 0.2, 0.08))

plot_f(1, ["S_0"], 50, 150)
plot_f(1, ["K"], 50, 150)
plot_f(1, ["r"], 0.01, 1)
plot_f(1, ["sigma"], 0.01, 1)
plot_f(2, ["M", "K"], 75, 125, K = 80)
plot_f(2, ["M", "K"], 75, 125, K = 100)
plot_f(2, ["M", "K"], 75, 125, K = 110)

