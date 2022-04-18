"""Generic Ransac estimator

Authors: Ayush Baid
"""
import math

from typing import Any, Callable, Tuple


class RansacInput:
    def __init__(self, data: Tuple[Any, ...], min_sample_size: int) -> None:
        self._input_length = len(data)
        self._data = data
        self._min_sample_size = min_sample_size

    def sample(self) -> Tuple[Any, ...]:
        pass


class RansacEstimator:
    def __init__(
        self,
        ransac_input: RansacInput,
        desired_confidence: float,
        model_fit_on_min_sample_fn: Callable[[Tuple[Any, ...]], Any],
        model_evaluation_fn: Callable[[Tuple[Any, ...], Any], float],
    ) -> None:
        self._ransac_input: RansacInput = ransac_input
        self._model_fit_on_min_sample_fn = model_fit_on_min_sample_fn
        self._model_evaluation_fn = model_evaluation_fn
        self._best_score = math.inf
        self._best_model = None

    def __run_iteration(self):
        # estimate hypothesis on the minimum sample size and score it using all the data
        sample = self._ransac_input.sample()
        hypothesis = self._model_fit_on_min_sample_fn(sample)
        score, inlier_flag = self._model_evaluation_fn(self._ransac_input, hypothesis)

        if score > self._best_score:
            self._best_score = score
            # TODO: fit the model on all the inliers using inlier_flag
            self._best_model = hypothesis

    def estimate(self):
        max_iters = self.__compute_max_iters()
        for _ in range(max_iters):
            self.__run_iteration()

        return self._best_model

    def __compute_max_iters(self):
        return 100
