# this is a script that converts a range of configuration files into individual configuration files that specify a single point in configuration space

import sys, os, itertools, copy
from utils.ConfigFileSweeper.TextFragment import TextFragment
from utils.ConfigFileSweeper.SweepDimension import SweepDimension
from utils.ConfigFileSweeper.FragmentParser import FragmentParser

def ConfigFileSweeper(infile_path, output_dir):

    infile_name = os.path.basename(infile_path)
    infile_basename, file_extension = os.path.splitext(infile_name)

    outfile_basename = infile_basename

    sds = {}

    # here, need to parse the file and generate the TextFragments
    # as well as their ordering
    with open(infile_path, 'r') as infile:
        fragments = FragmentParser(infile)
        for name, fragment in fragments:
            if name not in sds:
                # this fragment belongs to a SweepDimension that does not yet exist, create it
                sd = SweepDimension()
                sd.add_iterable(fragment)
                sds[name] = sd
                print("added new fragment with name '" + name + "'")
            else:
                # this fragment belongs to an already existing SweepDimension, hence add it as new iterable
                sds[name].add_iterable(fragment)
                print("reused existing fragment with name '" + name + "'")
    
    # get the actual list of SweepDimensions
    sdlist = sds.values()

    outfile_cnt = 0
    for cur in itertools.product(*sdlist):
        # compose the current version of the document by inserting
        # the current values of the individual text fragments at their
        # corresponding locations
        outlist = [None] * fragments.fragment_number

        for cur_sd in cur:
            for cur_fragment_pos, cur_fragment_val in cur_sd:
                outlist[cur_fragment_pos] = cur_fragment_val
                
        outfile_name = os.path.join(output_dir, outfile_basename + "_slice_" + str(outfile_cnt) + file_extension)

        with open(outfile_name, 'w') as outfile:
            outfile_cnt += 1

            for block_val in outlist:
                outfile.write(block_val)

if __name__ == "__main__":
    infile_path = sys.argv[1]

    if len(sys.argv) == 3:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.dirname(infile_path)

    ConfigFileSweeper(infile_path = infile_path, output_dir = output_dir)
