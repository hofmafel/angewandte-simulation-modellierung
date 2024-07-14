import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Black-Scholes formula for European Call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

# Parameters
S = 100  # Current stock price
r = 0.05  # Risk-free rate
sigma = 0.2  # Volatility

# Create a grid of strike prices and expiration times
strike_prices = np.linspace(50, 150, 50)
expiration_times = np.linspace(0.01, 2, 50)
strike_prices, expiration_times = np.meshgrid(strike_prices, expiration_times)

# Calculate option prices
option_prices = black_scholes_call(S, strike_prices, expiration_times, r, sigma)

# Plotting the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(strike_prices, expiration_times, option_prices, cmap='viridis')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Expiration Time')
ax.set_zlabel('Option Price')
ax.set_title('Option Price Surface')

plt.show()
