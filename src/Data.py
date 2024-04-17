"""
File by Samuel Kwiatkowski-Martin
This file is our data class, which will contain and process columns and rows
"""
import random  # for our shuffling
from Row import Row  # Imports the Row class from the Row file
from util import csv, rnd  # Imports the csv function from util
from Cols import Cols  # Imports the Cols class
from Node import Node  # Imports the Node class
from the import the


class Data:
    def __init__(self, src, func=None):
        """
        Initialization function for the data class
        Sets up rows and columns and calls to read in required values
        :param src: Either the csv filename or a list of rows
        :param func: An anonymous function, likely the function learn as of right now
        """
        self.rows = []
        self.cols = None
        if isinstance(src, str):
            # checks if src is a string
            self.process_file(src, func)
        ## else the scenario where source is a table already
        else:
            for x in src:
                self.add(x, func)

    def process_file(self, src, func=None):
        """
        Function that process the src file
        :param src: the csv file to be processed
        :param func: The anonymous function to continue being passed through
        :return: None
        """
        for row in csv(src):
            self.add(row, func)

    def add(self, t, func=None):
        """
        Function that adds rows and columns from the read csv
        :param t: t is the row to be added and should be passed as an array
        :param func: The anonymous function to be ran on each row before being added to a col
        :return: None
        """
        row = t if hasattr(t, 'cells') else Row(t)
        if self.cols is None:
            self.cols = Cols(row)
            # the following is not included in his lua code, but I believe we need it
            # self.rows.append((self.cols))
        else:
            if func is not None:
                # if the lambda function was passed to data
                func(self, row)
            self.rows.append(self.cols.add(row))

    def stats(self, cols="y", fun="mid", ndivs=2):
        """
        Computes the requested stats for whichever column requested
        :param cols: A str that is the cols we would like stats on
        :param fun: A str that determines which stat you would like, mean/standard deviation
        :param ndivs: An int giving us the number of precsion points in our returned stats
        :return: A dictionary that holds column names as keys, and stat values as values
        """
        u = {}  # a dictionary to be returned
        u[".N"] = len(self.rows) - 1
        if cols == "y":
            if fun == "mid":
                for col in self.cols.y:
                    #  Remember, col here is either a NUM or SYM object
                    u[col.txt] = round(col.mid(), ndivs)
            elif fun == "div":
                for col in self.cols.y:
                    u[col.txt] = round(col.div(), ndivs)
            elif fun == "small":
                pass  # doesn't need to be implemented yet
        elif cols == "x":
            if fun == "mid":
                for col in self.cols.x:
                    #  Remember, col here is either a NUM or SYM object
                    u[col.txt] = round(col.mid(), ndivs)
            elif fun == "div":
                for col in self.cols.x:
                    u[col.txt] = round(col.div(), ndivs)
            elif fun == "small":
                pass  # doesn't need to be implemented yet
        elif cols == "all":
            if fun == "mid":
                for col in self.cols.all:
                    #  Remember, col here is either a NUM or SYM object
                    u[col.txt] = round(col.mid(), ndivs)
            elif fun == "div":
                for col in self.cols.all:
                    u[col.txt] = round(col.div(), ndivs)
            elif fun == "small":
                pass  # doesn't need to be implemented yet
        else:
            #  This is our strange scenario where the str input in lets say an individual column
            if fun == "mid":
                for col in self.cols.all:
                    if col.txt in cols:
                        #  Remember, col here is either a NUM or SYM object
                        u[col.txt] = round(col.mid(), ndivs)
            elif fun == "div":
                for col in self.cols.all:
                    if col.txt in cols:
                        u[col.txt] = round(col.div(), ndivs)
            elif fun == "small":
                pass  # doesn't need to be implemented yet
        return u

    def mid(self, cols="all"):
        """
        Function returns the means and modes of all or a selection of columns
        :param cols: A str that tells the function which columns to calculate on
        :return: Return an array that represents the means/modes of whatever columns were selected
        """
        u = []  # an array to be returned

        if cols == "y":
            for col in self.cols.y:
                #  Remember, col here is either a NUM or SYM object
                u.append(col.mid())

        elif cols == "x":
            for col in self.cols.x:
                #  Remember, col here is either a NUM or SYM object
                u.append(col.mid())

        elif cols == "all":
            for col in self.cols.all:
                #  Remember, col here is either a NUM or SYM object
                u.append(col.mid())

        else:
            #  This is our strange scenario where the str input in lets say an individual column
            for col in self.cols.all:
                if col.txt in cols:
                    #  Remember, col here is either a NUM or SYM object
                    u.append(col.mid())

        return Row(u)

    def div(self, cols="all"):
        """
        Function returns the entropy/standard deviation of all or a selection of columns
        :param cols: A str that tells the function which columns to calculate on
        :return: Return an array that represents the sds/entropys of whatever columns were selected
        """
        u = []  # an array to be returned

        if cols == "y":
            for col in self.cols.y:
                #  Remember, col here is either a NUM or SYM object
                u.append(col.div())

        elif cols == "x":
            for col in self.cols.x:
                #  Remember, col here is either a NUM or SYM object
                u.append(col.div())

        elif cols == "all":
            for col in self.cols.all:
                #  Remember, col here is either a NUM or SYM object
                u.append(col.div())

        else:
            #  This is our strange scenario where the str input in lets say an individual column
            for col in self.cols.all:
                if col.txt in cols:
                    #  Remember, col here is either a NUM or SYM object
                    u.append(col.div())

        return Row(u)

    def clone(self, rows=[]):
        new = Data([self.cols.names])
        for row in rows:
            new.add(row)
        return new

    def split(self, best, rest, lite, dark):
        selected = Data([self.cols.names])
        max = -1E30
        out = 0
        for (i, row) in enumerate(dark):
            b = row.like(best, len(lite), 2)
            r = row.like(rest, len(lite), 2)
            if b > r:
                selected.add(row)
            tmp = abs(b + r) / abs(b - r + 1E-300)
            if tmp > max:
                out, max = i, tmp
        return out, selected

    def best_rest(self, rows, want):
        """
        This function divides the row as best and rest
        """
        rows = sorted(rows, key=lambda x: x.d2h(self))
        best = [self.cols.names]
        rest = [self.cols.names]
        for i, row in enumerate(rows):
            if i < want:
                best.append(row)
            else:
                rest.append(row)
        return Data(best), Data(rest)

    def baseline1_2(self, rows, top_num, option_num):
        """
        Prints the top X number of row's independent variable values
        :param rows: the shuffled rows object
        :param top_num: the total number of rows to have their y values printed
        :param option_num: just the num to be printedwws
        """
        i_var_indices = []  # list for the independent variable indices
        i_var_txts = []  # list of independent variable names
        for i_var in self.cols.y:
            # iterating over the different independent variables
            i_var_indices.append(i_var.at)
            i_var_txts.append(i_var.txt)
        print(f"{option_num}. top{top_num}", end=' ')
        for name in i_var_txts:
            print(f"{name},", end=' ')
        print()
        for i in range(top_num):
            print(f"Row {i + 1}", end=' ')
            for j in range(len(i_var_indices)):
                print(f"{rows[i].cells[i_var_indices[j]]}", end=' ')
            print()

    def baseline3(self, closest_row):
        """
        This function prints the y values of the first row after sorting by d2h
        :param closest_row: The closest row to the heaven point
        """
        row = closest_row
        i_var_indices = []  # list for the independent variable indices
        i_var_txts = []  # list of independent variable names
        for i_var in self.cols.y:
            # iterating over the different independent variables
            i_var_indices.append(i_var.at)
            i_var_txts.append(i_var.txt)
        print(f"3. most ")
        for name in i_var_txts:
            print(f"{name},", end=' ')
        print()
        print(f"Row {1}", end=' ')
        for j in range(len(i_var_indices)):
            # print the y values of row
            print(f"{row.cells[i_var_indices[j]]}", end=' ')
        print()

    def baseline4(self, budget0, budget_number, dark):
        """
        This function prints the y values of the centroid row of the selected
        :param budget0: initial number of rows to be evaluated
        :param budget_number: The number of budget in which it is printed
        :param dark: The dark rows
        """
        shuffled_row = random.sample(dark, len(dark))  # shuffles dark
        random_rows = [self.cols.names]
        for i in range(budget0 + budget_number):
            random_rows.append(shuffled_row[i])
        row = Data(random_rows).mid()
        i_var_indices = []  # list for the independent variable indices
        i_var_txts = []  # list of independent variable names
        for i_var in self.cols.y:
            # iterating over the different independent variables
            i_var_indices.append(i_var.at)
            i_var_txts.append(i_var.txt)
        print(f"4. rand")
        for name in i_var_txts:
            print(f"{name},", end=' ')
        print()
        for j in range(len(i_var_indices)):
            # print the y values of row
            print(f"{rnd(row.cells[i_var_indices[j]])}", end=' ')
        print()

    def baseline5(self, budget_number, selected_row):
        """
        This function prints the y values of the centroid row of the selected
        :param budget_number: The number of budget in which it is printed
        :param selected_row: The centroid row of SELECTED
        """
        row = selected_row
        i_var_indices = []  # list for the independent variable indices
        i_var_txts = []  # list of independent variable names
        for i_var in self.cols.y:
            # iterating over the different independent variables
            i_var_indices.append(i_var.at)
            i_var_txts.append(i_var.txt)
        print(f"5. mid")
        for name in i_var_txts:
            print(f"{name},", end=' ')
        print()
        for j in range(len(i_var_indices)):
            # print the y values of row
            print(f"{rnd(row.cells[i_var_indices[j]])}", end=' ')
        print()

    def baseline6(self, budget_number, best_row):
        """
        This function prints the y values of the first row in best
        :param budget_number: The number of budget in which it is printed
        :param best_row: The best row so far
        """
        row = best_row
        i_var_indices = []  # list for the independent variable indices
        i_var_txts = []  # list of independent variable names
        for i_var in self.cols.y:
            # iterating over the different independent variables
            i_var_indices.append(i_var.at)
            i_var_txts.append(i_var.txt)
        print(f"6. top")
        for name in i_var_txts:
            print(f"{name},", end=' ')
        print()
        for j in range(len(i_var_indices)):
            # print the y values of row
            print(f"{rnd(row.cells[i_var_indices[j]])}", end=' ')
        print()

    def gate(self, budget0, budget, some, print_baselines=True, test_name=""):
        """
        This function guesses, accesses, transforms the data, and then evaluates
        :param budget0: initial number of rows to be evaluated
        :param budget: the number of rows to subsequently evaluate
        :param some: a constant float value to determine how many rows to place in best versus rest
        :param test_name: the name of the test we are running if at all
        """
        stats = []
        bests = []
        if print_baselines:
            # Only print the baselines if we need to
            rows = random.sample(self.rows, len(self.rows))  # shuffles rows
            self.baseline1_2(rows, 6, 1)  # baseline #1
            self.baseline1_2(rows, 50, 2)  # baseline #2

            # Now we must sort rows based on the distance to heaven -- will fix this once d2h is done
            rows.sort(key=lambda x: x.d2h(self))
            # print some stuff...
            self.baseline3(rows[0])  # baseline 3

        rows = random.sample(self.rows, len(self.rows))  # reshuffle rows
        if test_name == "shuffle":
            # check and make sure the shuffle worked
            assert rows[0] != self.rows[0] and rows[-1] != self.rows[-1]
        lite = rows[0:budget0]  # grab first budget0 amount of rows
        dark = rows[budget0:]  # grab the remaining rows

        for i in range(budget):
            best, rest = self.best_rest(lite,
                                        len(lite) ** some)  # sort our known rows into good vs bad
            todo, selected = self.split(best, rest, lite,
                                        dark)  # figuring out which row is the most
            # confusing --> will be the index of the MOST confusing value while selected
            # is a data object storing the rows from dark that most liked best(of which were tested)

            # ngl I'm not sure what the point of the following 2 lines is
            stats.append(selected.mid())
            bests.append(best.rows[0])
            if print_baselines:
                # again only print baselines if we need to
                print(f"Budget {i} :", end='\n')
                self.baseline4(budget0, i, dark)
                self.baseline5(i, selected.mid())
                self.baseline6(i, best.rows[0])

            # Insert into the lite, the most confusing example from dark(also remove the val from
            # dark
            lite.append(dark.pop(todo))
        return stats, bests

    def farapart(self, rows, sortp=None, a=None):
        far = int((len(rows) * the.Far) // 1)
        evals = a and 1 or 2
        a = a or random.choice(rows).neighbors(self, rows)[far]
        b = a.neighbors(self, rows)[far]
        if sortp and b.d2h(self) < a.d2h(self):
            a, b = b, a
        return (a, b, a.dist(b, self), evals)

    def half(self, rows, sortp=None, before=None):
        some = random.sample(rows, min(the.Half, len(rows)))
        a, b, C, evals = self.farapart(some, sortp, before)

        def d(row1, row2):
            return row1.dist(row2, self)

        def project(r):
            return (d(r, a) ** 2 + C ** 2 - d(r, b) ** 2) / (2 * C) if C != 0 else float("inf")

        _as, _bs = [], []
        rows = rows.copy()
        rows.sort(key=project)
        for (n, row) in enumerate(rows):
            if n < len(rows) // 2:
                _as.append(row)
            else:
                _bs.append(row)
        return (_as, _bs, a, b, C, d(a, _bs[0]), evals)

    def tree(self, sortp):
        """
        Recursive random projects. 'Half' then data, then recurse on each half.
        :param sortp: Boolean, true if sorted? I think...
        """
        evals = 0

        def _tree(data, above=None):
            """
            Recursive tree function
            :param data:
            :param above:
            """
            node = Node(data)
            if len(data.rows) > 2 * (len(self.rows)) ** .5:
                lefts, rights, node.left, node.right, node.C, node.cut, evals1 = self.half(
                    data.rows, sortp, above)
                nonlocal evals
                evals = evals + evals1
                node.lefts = _tree(self.clone(lefts), node.left)
                node.rights = _tree(self.clone(rights), node.right)
            return node

        return _tree(self), evals

    def branch(self, stop=None):
        evals, rest = 1, []
        stop = stop or (2 * (len(self.rows)) ** 0.5)

        def _branch(data, above=None):
            nonlocal evals
            if len(data.rows) > stop:
                lefts, rights, left = self.half(data.rows, True, above)[:3]
                evals += 1
                for row in rights:
                    rest.append(row)
                return _branch(data.clone(lefts), left)
            else:
                return self.clone(data.rows), self.clone(rest), evals

        return _branch(self)

