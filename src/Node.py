"""
File by Samuel Kwiatkowski-Martin
This file is our Node class, which will be used to help with recursive random projection
"""
import math # for testing purposes


class Node:
    def __init__(self, data):
        self.here = data
        # Declare our class variables to be edited later
        self.left = None
        self.right = None
        self.C = None
        self.cut = None
        self.lefts = None
        self.rights = None

    def walk(self, fun, depth=0):
        """
        Walks through a tree and executes whatever function is passed at each Node
        :param fun:  the anonymous function to be executed at each node
        :param depth: the current node we are at's depth
        """
        fun(self, depth, not (self.lefts or self.rights))  # send a boolean saying whether we are at
        # a leaf value or not --> this fun will likely be a printing statement in our context
        if self.lefts:
            #  if there is a node to our bottom left then keep walking the walk
            self.lefts.walk(fun, depth+1)
        if self.rights:
            self.rights.walk(fun, depth+1)

    def show(self):
        """
        Actually prints each node that this function is called on
        """
        def d2h(data):
            """
            A local function which calculates the param "data"'s mid, and then calculates
            the distance to heaven from that mid to self.here(the data object which initially built
            the Node class
            :param data: the data object we are using to determine the heaven point
            """
            return round(data.mid().d2h(self.here), 2)

        maxDepth = 0
        def _show(node, depth, leafp):
            """
            Function for literally printing the depth we are at, and the associated mid value's row
            :param node: Node, the current node we are at
            :param depth: Int, the current depth we are at in our tree
            :param leafp: Boolean, true if we are at a leaf node, else false
            """
            print_cells = node.here.mid().cells
            for i in range(len(print_cells)):
                print_cells[i] = round(print_cells[i], 2)
            post = f"{d2h(node.here)} \t{print_cells}" if leafp else f""
            nonlocal maxDepth  ## asscoiates this maxDepth with the maxDepth above the _show
            maxDepth = max(maxDepth, depth)
            print(f"{'|.. '*depth}{post}")  # print it out!
        self.walk(_show)
        # test that the max depth of recursive tree doesn't go above log2(N) where N is
        # the number of data points
        assert maxDepth <= math.log2(len(self.here.rows))
        print("")
        print_cells = self.here.mid().cells
        for i in range(len(print_cells)):
            print_cells[i] = round(print_cells[i], 2)  # round all values by 2 decimal places
        print(f"{'    '*maxDepth} {d2h(self.here)} {print_cells}")
        print(f"{'    '*maxDepth} ---- {self.here.cols.names}")

