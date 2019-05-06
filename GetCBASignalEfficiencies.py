from argparse import ArgumentParser
from base.Configs import TrainingConfig
from analysis.CutBasedCategoryFiller import CutBasedCategoryFiller
from DatasetExtractor import TrainNuisAuxSplit

def GetCBASignalEfficiencies():
    

if __name__ == "__main__":
    parser = ArgumentParser(description = "optimize the cuts in the CBA for maximum binned significance")
    args = vars(parser.parse_args())

    outdir = args["outdir"]
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    effs = GetCBASignalEfficiencies(**args)
    print(effs)
