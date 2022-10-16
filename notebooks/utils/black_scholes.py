import numpy as np
from scipy.stats import norm


class BlackScholesOptionPrice():
    """Class for Black-Scholes price of European put and call options."""

    def __init__(self, strike: float, r: float, sigma: float):
        
        self.r = r
        self.sigma = sigma
        self.strike = strike

    def _d1_and_d2(self, current_stock_price, time_to_maturity):
        """Calculates auxiliary d_1 and d_2 which enter the N(0, 1) cdf in the pricing formulas"""
        
        d1 = (np.log(current_stock_price / self.strike) + (self.r + 0.5 * self.sigma ** 2) * time_to_maturity) / (self.sigma * np.sqrt(time_to_maturity))
        d2 = d1 - self.sigma * np.sqrt(time_to_maturity)
        return d1, d2

    def price_put(self, current_stock_price, time_to_maturity):
        """Calculates price of European put option"""

        d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
        return np.exp(-self.r * time_to_maturity) * self.strike * norm.cdf(-d2) - current_stock_price * norm.cdf(-d1)
    
    def delta_put(self, current_stock_price, time_to_maturity):
        """Calculates delta of European put option."""
        
        d1, _ = self._d1_and_d2(current_stock_price, time_to_maturity)
        return - norm.cdf(- d1)
                
    def price_call(self, current_stock_price, time_to_maturity):
        """Calculates price of European call option"""

        d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
        return current_stock_price * norm.cdf(d1) - np.exp(-self.r * time_to_maturity) * self.strike *  norm.cdf(d2)
    
    def delta_call(self, current_stock_price, time_to_maturity):
        """Calculates delta of European call option."""
        
        d1, _ = self._d1_and_d2(current_stock_price, time_to_maturity)
        return norm.cdf(d1)
    
    def _vega(self, current_stock_price, time_to_maturity):
        """Computes vega."""
        
        d1, _ = _d1_and_d2(current_stock_price, time_to_maturity)
        return current_stock_price * norm.pdf(d1) * np.sqrt(time_to_maturity)
    
    def vega_call(self, current_stock_price, time_to_maturity):
        return self._vega(current_stock_price, time_to_maturity)

    def vega_put(self, current_stock_price, time_to_maturity):
        return self._vega(current_stock_price, time_to_maturity)

