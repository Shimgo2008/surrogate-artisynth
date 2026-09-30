from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self, params: list, grads: list):
        pass
