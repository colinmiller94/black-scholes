import math
from scipy.stats import norm


class Option:
    """
    Base class for Call and Put sub classes. Using Black-Scholes model from Sheldon Natenurg's Option Volatility and
    Pricing book

    :param float spot: Spot price of underlying
    :param float strike: Strike price of option
    :param float tte: time to expiration (years)
    :param float int_rate: interest rate
    :param float b: see Natenburg. Not exactly sure what best to call this. Same as interset rate for options on stock
    :param float vol: volatility
    """

    def __init__(self, spot, strike, tte, int_rate, b, vol):
        self.spot = spot
        self.strike = strike
        self.tte = tte
        self.int_rate = int_rate
        self.b = b
        self.vol = vol

        # These attributes will be calculated in the Call/Put subclasses, as they are different for each type
        self.theo = self.delta = self.theta = None

        self._update_shared_greeks()

    def _update_shared_greeks(self):
        # calculate some intermediate terms that apply to both calls and puts
        self._d1 = (math.log(self.spot / self.strike) + (self.b + self.vol ** 2 / 2.0) * self.tte) / (
                self.vol * math.sqrt(self.tte))
        self._d2 = self._d1 - self.vol * math.sqrt(self.tte)

        self._theta_term1 = -self.spot * math.exp((self.b - self.int_rate) * self.tte) * norm.pdf(
            self._d1) * self.vol / (2 * math.sqrt(self.tte))
        self._theta_term2 = (self.b - self.int_rate) * self.spot * math.exp(
            (self.b - self.int_rate) * self.tte) * norm.cdf(self._d1)

        # Gamma and Vega are equivalent for calls and puts
        self.gamma = math.exp((self.b - self.int_rate) * self.tte) * norm.pdf(self._d1) / (
                self.spot * self.vol * math.sqrt(self.tte))

        self.vega = (self.spot * math.exp((self.b - self.int_rate) * self.tte) * norm.pdf(self._d1) * math.sqrt(
            self.tte))


class Call(Option):
    """
    Inherits from Option class. Same arguments
    """

    def __init__(self, spot, strike, tte, int_rate, b, vol):
        super().__init__(spot, strike, tte, int_rate, b, vol)
        self.update_greeks()

    def update_greeks(self):
        self._update_shared_greeks()

        self.theo = self.spot * math.exp((self.b - self.int_rate) * self.tte) * norm.cdf(
            self._d1) - self.strike * math.exp(-self.int_rate * self.tte) * norm.cdf(self._d2)

        self.delta = math.exp((self.b - self.int_rate) * self.tte) * norm.cdf(self._d1)
        self.theta = (self._theta_term1 - self._theta_term2 - self.int_rate * self.strike * math.exp(
            -self.int_rate * self.tte) * norm.cdf(self._d2))


class Put(Option):
    """
    Inherits from Option class. Same arguments
    """

    def __init__(self, spot, strike, tte, int_rate, b, vol):
        super().__init__(spot, strike, tte, int_rate, b, vol)
        self.update_greeks()

    def update_greeks(self):
        self._update_shared_greeks()

        self.theo = self.strike * math.exp(-self.int_rate * self.tte) * norm.cdf(
            -self._d2) - self.spot * math.exp((self.b - self.int_rate) * self.tte) * norm.cdf(-self._d1)

        self.delta = math.exp((self.b - self.int_rate) * self.tte) * (norm.cdf(self._d1) - 1)

        self.theta = (self._theta_term1 + self._theta_term2 + self.int_rate * self.strike * math.exp(
            -self.int_rate * self.tte) * norm.cdf(-self._d2))
