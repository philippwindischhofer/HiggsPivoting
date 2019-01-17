from abc import ABC, abstractmethod

class ClassifierModel(ABC):

    @abstractmethod
    def build_model(self, in_tensor):
        pass

    @abstractmethod
    def build_loss(self, pred_tensor, label_tensor):
        pass

class AdversaryModel(ABC):
    
    @abstractmethod
    def build_loss(self, pred, nuisance):
        pass
