from abc import ABC, abstractmethod

from torch import Tensor


class DecisionFunction(ABC):

    @abstractmethod
    def __call__(self, z: Tensor, p: Tensor, i_chosen: int,
                 v_chosen: int) -> int:
        pass
