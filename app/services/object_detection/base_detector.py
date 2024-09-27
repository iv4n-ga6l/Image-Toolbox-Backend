from abc import ABC, abstractmethod

class BaseObjectDetector(ABC):
    @abstractmethod
    def detect_objects(self, image):
        pass

    @abstractmethod
    def count_objects(self, image):
        pass