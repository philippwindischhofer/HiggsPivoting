class SweepDimension:

    def __init__(self, itlist = None):
        if itlist is not None:
            self.itlist = itlist
        else:
            self.itlist = []

    def add_iterable(self, it):
        self.itlist.append(it)

    def iter(self):
        return self.__iter__()

    def __iter__(self):
        return self

    def next(self):
        # for each block in the itlist, return the next element
        retval = [block.next() for block in self.itlist]

        return retval
