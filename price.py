import math
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European call option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    
    Returns:
    float: Call option price
    """
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price for a European put option.
    
    Parameters:
    S (float): Current stock price
    K (float): Strike price of the option
    T (float): Time to maturity in years
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    
    Returns:
    float: Put option price
    """
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return put_price

# Example usage:
S = 100    # Current stock price
K = 100    # Strike price
T = 1      # Time to maturity in years
r = 0.05   # Risk-free interest rate (5%)
sigma = 0.2 # Volatility (20%)

call_price = black_scholes_call(S, K, T, r, sigma)
put_price = black_scholes_put(S, K, T, r, sigma)

print(f"The Black-Scholes price for the European call option is: {call_price:.2f}")
print(f"The Black-Scholes price for the European put option is: {put_price:.2f}")
