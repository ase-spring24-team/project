"""
File by Sai Raj Thirumalai
This file is our Sym class, which will contain the Sym data
"""

import math
from the import the

class Sym:
    def __init__(self, s=" ", n=0):
        """
        Initialization function for the Sym class
        Sets up the column name, column index, number of rows and data aggregates
        :param s: column name, n: column index
        """
        self.txt = s
        self.at = n
        self.n = 0 #  testing
        self.has = {}
        self.mode = None
        self.most = 0

    def add(self, x):
        """
        Function that adds a value
        :param x: data value
        """
        if x != "?": # ignore if value is ?
            self.n += 1
            # increase the count of value
            if x in self.has:
                self.has[x] += 1
            else:
                self.has[x] = 1
            # update mode value if needed
            if self.has[x] > self.most:
                self.most = self.has[x]
                self.mode = x

    def mid(self):
        """
        Function that returns the mid (mode) value
        :return: mean
        """
        return self.mode

    def div(self):
        """
        Function that returns the standard deviation
        :return: standard deviation
        """
        e = 0
        for _, v in self.has.items():
            e -= v / self.n * math.log(v / self.n, 2)
        return e
    
    def like(self, x, prior):
        h = self.has[x] if x in self.has else 0
        if self.n == 0 and the.m == 0:
            return 0
        else:
            return (h + the.m * prior)/ (self.n + the.m)

    def dist(self, x, y):
        """
        This function returns the distance of a sym object to another sym object
        However, sym objects aren't numbers so we assume the distance is 1 in every case where
        they aren't the same, or where we don't know one of the value. Otherwise, they are the same
        meaning 0 distance
        :param x: the self row's sym value
        :param y: the row we are comparing to's sym value
        """
        if x == "?" or y == "?":
            # if either values is unknown
            return 1  # return a maximum distance of 1
        return 1 if x != y else 0

    def bin(self, x):
        """
        Function that discretizes a Sym, however, symbols are binned by exactly what they are
        unlike numbers, so just return the sym
        """
        return x
