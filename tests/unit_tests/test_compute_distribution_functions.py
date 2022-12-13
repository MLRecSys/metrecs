from metrecs.utils import (
    compute_distribution_multiple_categories,
    compute_distribution,
    harmonic_number,
)
import numpy as np


def test_compute_distribution():
    a = np.array(["a", "b", "c", "c"])
    weights_radio = np.array(
        [1 / rank / harmonic_number(len(a)) for rank in range(1, len(a) + 1)]
    )
    assert compute_distribution(a, weights=weights_radio) == {
        "a": 0.4799997900047559,
        "b": 0.23999989500237795,
        "c": 0.2799998775027743,
    }
    assert compute_distribution(a) == {"a": 0.25, "b": 0.25, "c": 0.5}


compute_distribution_multiple_categories(a=np.array([["x", "a"], ["a", "x"], ["a"]]))

compute_distribution(["x", "a"], weights=[1, 0.5, 5])


def test_compute_distribution_multiple_categories():
    pass
