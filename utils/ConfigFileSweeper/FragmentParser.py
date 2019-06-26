from utils.ConfigFileSweeper.SliceTextFragment import SliceTextFragment
from utils.ConfigFileSweeper.TextFragment import TextFragment
import re

class FragmentParser:

    available_types = [SliceTextFragment] # may want to extend this for additional functionality
    
    def __init__(self, infile):
        self.infile = infile
        self.fragment_number = 0
        self.line = self.infile.readline()

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        fragment_name = ""
        state = "read_header_line"
        line_buffer = []

        # go through the file until find the next complete fragment, then construct and return it
        while self.line:

            if state == "read_header_line":

                state = "read_fragment_body"

                # expect this line to be of the header type, otherwise have an unnamed (and inactive)
                # fragment here
                fragment_type, fragment_name = self._get_fragment_type(self.line)

                # in case this is an unnamed fragment, avoid losing this line, which is already part of the body of the fragment
                if not fragment_name:
                    fragment_name = "unnamed"
                else:
                    self.line = self.infile.readline()
                
            elif state == "read_fragment_body":

                if not self._is_control_line(self.line):
                    # keep appending the lines within the fragment
                    line_buffer.append(self.line)
                    self.line = self.infile.readline()
                elif fragment_type is not None:
                    # have reached the end of the fragment, instantiate the fragment
                    state = "read_footer_line"
                else:
                    # have found the end of the current unnamed fragment and the beginning of a new fragment
                    # therefore, instantiate the current unnamed fragment and return it
                    retval = TextFragment(self.fragment_number, ["".join(line_buffer)])
                    self.fragment_number += 1
                    return fragment_name, retval

            elif state == "read_footer_line":
                retval = fragment_type(self.fragment_number, line_buffer, self.line)
                self.fragment_number += 1

                self.line = self.infile.readline()
                return fragment_name, retval

        # could still have an 'unnamed' fragment open at this point!
        if fragment_name:
            retval = TextFragment(self.fragment_number, ["".join(line_buffer)])
            self.fragment_number += 1
            return fragment_name, retval

        raise StopIteration

    # checks if this line starts a new (named) fragment
    def _get_fragment_type(self, line):
        # go through the list of supported fragment types and see if any of them match
        for cur_type in FragmentParser.available_types:
            name = cur_type.check_title_line(line)
            if name:
                return cur_type, name

        return None, None # this line does not start any new named fragment

    def _is_control_line(self, line):
        control_re = re.compile("#\/\/CFS.*")

        return True if control_re.match(line) else False
