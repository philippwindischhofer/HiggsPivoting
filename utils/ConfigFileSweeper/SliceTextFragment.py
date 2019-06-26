# this is a more special kind of text fragment, namely a text fragment that can slice itself into several lines per slice
from utils.ConfigFileSweeper.TextFragment import TextFragment
import re

class SliceTextFragment(TextFragment):
    id_string = "START_SLICE"

    def __init__(self, pos, lines_in, options_line):

        def chunks(inlist, per_slice):
            splits = list(range(0, len(inlist), per_slice))
            if splits[-1] != len(inlist):
                splits.append(len(inlist))

            for i in range(len(splits) - 1):
                yield inlist[splits[i]:splits[i + 1]]

        per_slice = self._parse_options_line(options_line)

        fragment_entries = []
        for chunk in chunks(lines_in, per_slice):
            fragment_entries.append("".join(chunk))

        super(SliceTextFragment, self).__init__(pos, entries = fragment_entries)
    
    def _parse_options_line(self, line_in):
        per_slice_re = re.compile(".*PER_SLICE\((.*)\)")
        per_slice = int(per_slice_re.search(line_in).group(1))
        return per_slice

    # this method checks if the title line specifies a fragment that is
    # of this type, if so, it returns the name of the fragment
    @staticmethod
    def check_title_line(line_in):
        title_re = re.compile(".*START_SLICE\((.*)\)")
        names_list = title_re.findall(line_in)

        if names_list:
            return names_list[0]
        else:
            return None
