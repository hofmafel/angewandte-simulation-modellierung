import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Black-Scholes put option price function
def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

# Function to find implied volatility
def implied_volatility(S, K, T, r, market_price):
    objective = lambda sigma: (black_scholes_put(S, K, T, r, sigma) - market_price)**2
    result = minimize(objective, 0.2, bounds=[(1e-5, 2.0)])
    return result.x[0]

# Parameters
S = 468  # Current stock price
K = 470  # Strike price
T = 74 / 365  # Time to expiration in years
r = 0.04  # Risk-free rate
market_price = 18  # Current put option price

# Calculate implied volatility
sigma = implied_volatility(S, K, T, r, market_price)

# Generate a range of stock prices at expiration
stock_prices = np.linspace(300, 650, 500)
pdf = stats.norm.pdf((np.log(stock_prices / S) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)), 0, 1) / (stock_prices * sigma * np.sqrt(T))

# Calculate sigma ranges
mu = S * np.exp((r - 0.5 * sigma**2) * T)
sigma_1 = S * sigma * np.sqrt(T)
sigma_2 = 2 * sigma_1
sigma_3 = 3 * sigma_1

# Plot the probability distribution
plt.figure(figsize=(10, 6))
plt.plot(stock_prices, pdf, label=f'Probability Distribution\n(Sigma: {sigma:.2f})')

# Plot sigma lines
plt.axvline(mu + sigma_1, color='r', linestyle='--', label='1 Sigma (68.27%)')
plt.axvline(mu - sigma_1, color='r', linestyle='--')
plt.axvline(mu + sigma_2, color='g', linestyle='--', label='2 Sigma (95.45%)')
plt.axvline(mu - sigma_2, color='g', linestyle='--')
plt.axvline(mu + sigma_3, color='b', linestyle='--', label='3 Sigma (99.73%)')
plt.axvline(mu - sigma_3, color='b', linestyle='--')

# Add labels and legend
plt.title('Probability Distribution of Stock Prices at Expiration')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

