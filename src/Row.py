"""
File by Samuel Kwiatkowski-Martin
This file is our row class
"""
import math
from the import the
class Row:
    """
    Initializes the row using a passed array
    param t: the array the represents the row we care about
    """
    def __init__(self, t):
        self.cells = t

    def likes(self, datas):
        n, nHypotheses = 0, 0
        most = out = None
        for (k, data) in datas.items():
            n += len(data.rows)
            nHypotheses += 1
        for (k, data) in datas.items():
            tmp = self.like(data, n, nHypotheses)
            if most is None or tmp > most:
                most, out = tmp, k
        return out, most
    
    def like(self, data, n, nHypotheses):
        """
        Checks how much ROW like 'self'.
        :param data: The data class we are checking against to see how much the new row is like
        :param n: The total number of data points so far
        :param nHypotheses: The total number of classification options
        """
        prior = (len(data.rows) + the.k) / (n + the.k*nHypotheses)
        out = math.log(prior)
        for col in data.cols.x:
            #  where each col is either a col or sym object
            v = self.cells[col.at]
            if v != "?":
                #  if the column is supposed to be processed then
                inc = col.like(v,prior)
                out = out + (math.log(inc) if inc != 0 else float("-inf"))
        return math.exp(1)**out

    def d2h(self, data):
        """
        Function that returns the d2h (distance to heaven) value
        :return: d2h
        """
        d, n = 0, 0
        for col in data.cols.y:
            n += 1
            d += abs(col.heaven - col.norm(self.cells[col.at])) ** 2
        return math.sqrt(d) / math.sqrt(n)
    
    def dist(self, other, data):
        d, n = 0, 0
        p = the.p

        for col in data.cols.x :
            n += 1
            d += col.dist(self.cells[col.at], other.cells[col.at]) ** p
        
        return (d/n) ** (1/p)
    
    def neighbors(self, data, rows = None):
        if rows is None:
            rows = data.rows
        rows = rows.copy()
        rows.sort(key=lambda row: self.dist(row, data))
        return rows