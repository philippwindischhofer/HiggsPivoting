import os
from configparser import ConfigParser

from AdversarialClassifierEnvironment import AdversarialClassifierEnvironment
from MINEClassifierEnvironment import MINEClassifierEnvironment

class ConfigFileUtils:

    # attempt to load the config file and extract the model type
    @staticmethod
    def get_env_type(indir):
        model_dict = {"AdversarialClassifierEnvironment": AdversarialClassifierEnvironment,
                      "MINEClassifierEnvironment": MINEClassifierEnvironment}

        try:
            infile = os.path.join(indir, "meta.conf")
            gconfig = ConfigParser()
            gconfig.read(infile)
            model_type = gconfig["global"]["type"]

            return model_dict[model_type]
        except (FileNotFoundError, KeyError):
            return None
