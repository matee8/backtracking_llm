from abc import ABC, abstractmethod

from torch import Tensor


class DecisionFunction(ABC):

    @abstractmethod
    def __call__(self, logits: Tensor, probabilities: Tensor,
                 chosen_token_idx: int, chosen_token_id: int) -> int:
        pass
