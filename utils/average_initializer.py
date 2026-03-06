from typing import Union, List
import numpy as np

from GroupMultiNeSS import BaseMultiNeSS
from GroupMultiNeSS.utils import if_scalar_or_given_length_array, hard_thresholding_operator, fill_nan


class AverageMultiNeSS(BaseMultiNeSS):
    def __init__(self, d_shared: int = None, d_individs: Union[List[int], int] = None,
                 edge_distrib: str = "normal", loops_allowed: bool = True):
        super().__init__(edge_distrib=edge_distrib, loops_allowed=loops_allowed)
        self.d_shared = d_shared
        self.d_individs = d_individs

    def _validate_input(self, As: List[np.array]) -> np.ndarray:
        As = super()._validate_input(As)
        if self.d_individs is not None:
            self.d_individs_ = if_scalar_or_given_length_array(self.d_individs, length=self.n_layers_, name="d_individs")
        else:
            self.d_individs_ = [None] * self.n_layers_
        return As

    def fit(self, As: List[np.ndarray]):
        As = self._validate_input(As)
        self._init_param_matrices(shape=(1 + self.n_layers_, self.n_nodes_, self.n_nodes_))
        shared_comp = hard_thresholding_operator(np.nanmean(As, axis=0), max_rank=self.d_shared)
        individ_comps = [hard_thresholding_operator(fill_nan(A - shared_comp), max_rank=d_individ)
                         for A, d_individ in zip(As, self.d_individs_)]
        self._set_matrices([shared_comp, *individ_comps])
        return self
