from training.AdversarialModelTrainer import AdversarialModelTrainer

class ModelCollectionTrainer:

    def __init__(self, mcoll, batch_sampler):
        self.mcoll = mcoll
        self.batch_sampler = batch_sampler

    def train(self, trainsamples_sig, trainsamples_bkg, valsamples_sig, valsamples_bkg):

        print("have {} models to train".format(len(self.mcoll.models)))
        
        for ind, model in enumerate(self.mcoll.models):
            print("now training model {}".format(ind))

            trainer = AdversarialModelTrainer(model, self.batch_sampler, model.training_config)
            trainer.train(trainsamples_sig, trainsamples_bkg, valsamples_sig, valsamples_bkg)

