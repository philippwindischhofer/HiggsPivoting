# a text fragment has a certain, fixed position in the overall document

class TextFragment(object):
    
    def __init__(self, pos, entries = None):
        if entries is not None:
            self.entries = entries
        else:
            self.entries = []

        self.pos = pos
        self.ind = 0

    def add_entry(self, entry):
        self.entries.append(entry)

    def __iter__(self):
        return self

    def next(self):
        if self.ind < len(self.entries):
            retval = self.entries[self.ind]
            self.ind += 1
            return self.pos, retval
        else:
            raise StopIteration
