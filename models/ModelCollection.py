import pandas as pd
import numpy as np
import json, os
from configparser import ConfigParser

from models.AdversarialModel import AdversarialModel

class ModelCollection:

    def __init__(self, models):
        self.models = models
        self.default_value = [[-99, -99]]

    @classmethod
    def from_config(cls, config_dir):
        gconfig = ConfigParser()
        gconfig.read(os.path.join(config_dir, "meta.conf"))

        # extract the list of models that comprise this ModelCollection
        global_pars = gconfig["ModelCollection"]
        contained_models = global_pars["models"].split(',')

        print("found the following contained models:")
        print('\n'.join(contained_models))

        models = []

        for model_name in contained_models:
            
            model_dir = os.path.join(config_dir, model_name)

            # first need to prepare the initial folder structure for the model
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

                model_config = AdversarialModel.extract_config(model_name, gconfig)
                model_config_path = os.path.join(model_dir, "meta.conf")
                with open(model_config_path, 'w') as model_config_file:
                    model_config.write(model_config_file)
                
            model = AdversarialModel.from_config(model_dir)
            models.append(model)

        # create the ModelCollection
        return cls(models)

    def predict(self, data):
        assert isinstance(data, pd.DataFrame)

        preds = []
        for model in self.models:
            
            # get the chunk of data that this model requires
            model_data = model.data_formatter.format_as_TrainingSample(data).data

            # and also get the location of this chunk in the complete dataset
            model_indices = model.data_formatter.get_formatted_indices(data)
            
            # perform the prediction
            if len(model_data) > 0:
                model_pred = pd.DataFrame(model.predict(model_data), index = model_indices)
                preds.append(model_pred)

        # _something_ will have happened
        assert len(preds) > 0
        retval = pd.concat(preds).sort_index()
        
        # check if some data has not been seen by any model
        missed_indices = [ind for ind in data.index if ind not in retval.index]
        missed_retval = pd.DataFrame(np.repeat(self.default_value, len(missed_indices), axis = 0), 
                                     index = missed_indices)

        if len(missed_indices) > 0:
            print("Had no model available for {} entries!".format(len(missed_indices)))
            retval = pd.concat([retval, missed_retval]).sort_index()

        return retval.as_matrix()


