from abc import ABC, abstractmethod

class Preprocessor(ABC):

    @abstractmethod
    def process(self, data):
        pass
