from configparser import ConfigParser
import os

class TrainingConfig:
    # branches to use for the training
    training_branches = ["mBB", "dRBB", "pTB1", "pTB2", "MET", "dEtaBB", "dPhiMETdijet", "SumPtJet"]
    nuisance_branches = ["mBB"]
    auxiliary_branches = ["EventWeight"]

    training_pars = {"sow_target": 0.3, "pretrain_batches": 100, "training_batches": 800, "printout_interval": 10}

    @classmethod
    def from_file(cls, config_dir):
        gconfig = ConfigParser()
        gconfig.read(os.path.join(config_dir, "meta.conf"))

        cur_pars = {key: float(val) for key, val in gconfig["TrainingConfig"].items()}

        obj = cls()
        obj.training_pars.update(cur_pars)

        return obj        
