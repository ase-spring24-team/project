"""
File by Sathiya Narayanan Venkatesan
This file is our cols class, which will have all the column properties
"""

import re
from Num import Num
from Sym import Sym


class Cols:
    def __init__(self, row):
        """
        Initialization function for the cols class
        Sets up columns
        :param row: the column names in a the form of a array
        """
        self.x = []
        self.y = []
        self.all = []
        self.klass = None
        self.names = row.cells
        for idx, cell in enumerate(row.cells):
            col = Num(cell, idx) if re.match("^[A-Z]", cell) else Sym(cell, idx)
            self.all.append(col)
            if not cell.endswith("X"):
                if cell.endswith("!"):
                    self.klass = col
                if cell.endswith("!") or cell.endswith("+") or cell.endswith("-"):
                    self.y.append(col)
                else:
                    self.x.append(col)

    def add(self, row):
        """
        Function that adds rows
        :param row: row is the row to be added and should be passed as an Row object
        :return: None
        """
        for _, cols in enumerate([self.x, self.y]):
            for col in cols:
                col.add(row.cells[col.at])

        return row
