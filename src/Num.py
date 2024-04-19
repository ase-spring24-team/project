"""
File by Sai Raj Thirumalai
This file is our Num class, which will contain the numeric data
"""
import math
from the import the

class Num:
    def __init__(self, s=" ", n=0):
        """
        Initialization function for the Num class
        Sets up the column name, column index, number of rows and data aggregates
        :param s: column name, n: column index
        """
        self.txt = s
        self.at = n
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.hi = -1E30
        self.lo = 1E30
        self.heaven = 0 if s.endswith("-") else 1

    def none_fixer(self, x):
        """
        This function is designed to fix the none typo in random_forest
        :param x:
        :return:
        """
        if isinstance(x, str) or x is None:
            return 0  ## in our case this only occurs if None
        else:
            return x

    def add(self, x):
        """
        Function that adds a value
        :param x: data value
        """
        if x != "?": # ignore if value is ?
            self.n += 1
            # calculate mean value
            x = self.none_fixer(x)

            d = x - self.mu
            self.mu += d / self.n
            self.m2 += d * (x - self.mu)
            # update high and low values
            self.lo = min(x, self.lo)
            self.hi = max(x, self.hi)

    def mid(self):
        """
        Function that returns the mid (mean) value
        :return: mean
        """
        return self.mu

    def div(self):
        """
        Function that returns the standard deviation
        :return: standard deviation
        """
        if self.n < 2:
            return 0
        else:
            return (self.m2 / (self.n - 1)) ** 0.5

    def norm(self, x):
        """
        Function that returns the normalized value
        :param x: data value
        :return: normalized value
        """
        if x == "?":
            return x
        else:
            x = self.none_fixer(x)
            return (x - self.lo) / (self.hi - self.lo + 1E-30)

    # Likelihood
    def like(self, x, _):
        x = self.none_fixer(x)
        mu, sd = self.mid(), (self.div() + 1E-30)
        nom = 2.718 ** (-.5*(x - mu)**2 / (sd **2))
        denom = (sd*2.5 + 1E-30)
        return nom/denom

    def dist(self, x, y):
        """
        Function check the distance between one of self's row num values and some other row's
        num value
        :param x: This instances value at a specific column in the row
        :param y: The other row instance's value a specific column
        """
        if x == "?" and y == "?":
            # if both the x and y values are unknown, assume the worst
            return 1
        x, y = self.norm(x), self.norm(y)
        if x == "?":
            x = 1 if y < .5 else 0
        if y == "?":
            y = 1 if x < .5 else 0
        return abs(x-y)

    def bin(self, x):
        """
        Discretization of our NUM
        """
        x = self.none_fixer(x)
        tmp = (self.hi - self.lo)/(the.bins-1)
        return 1 if self.hi == self.lo else int(math.floor(x/tmp + .5)*tmp)

