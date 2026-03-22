from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch


class ModelIOStrategy(ABC):
    @abstractmethod
    def load(self, model, location):
        pass

    @abstractmethod
    def save(self, model, location):
        pass


class LossExtractionStrategy(ABC):
    @abstractmethod
    def extract_loss(self, x):
        pass


class ZeroGradStrategy(ABC):
    @abstractmethod
    def zero_grad(self, optimizer=None):
        pass


class DyNetModelIO(ModelIOStrategy):
    def load(self, model, location):
        model.model.populate(location)

    def save(self, model, location):
        model.model.save(location)


class PyTorchModelIO(ModelIOStrategy):

    def __init__(self, device):
        self.device = device

    def load(self, model, location):
        model.load_state_dict(torch.load(location, map_location=self.device))

    def save(self, model, location):
        torch.save(model.state_dict(), location)


class DyNetLossExtraction(LossExtractionStrategy):
    def extract_loss(self, x):
        return x.scalar_value()


class PyTorchLossExtraction(LossExtractionStrategy):
    def extract_loss(self, x):
        return x.item()


class DyNetZeroGrad(ZeroGradStrategy):
    def zero_grad(self, optimizer=None):
        # dy.renew_cg() is called explicitly in the training loop before each instance
        pass


class PyTorchZeroGrad(ZeroGradStrategy):
    def zero_grad(self, optimizer):
        optimizer.zero_grad()


@dataclass
class ModelContext:
    model_io: ModelIOStrategy
    loss_ext: LossExtractionStrategy
    zero_grad: ZeroGradStrategy


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
contextPytorch = ModelContext(PyTorchModelIO(DEVICE), PyTorchLossExtraction(), PyTorchZeroGrad())
contextDynet = ModelContext(DyNetModelIO(), DyNetLossExtraction(), DyNetZeroGrad())
