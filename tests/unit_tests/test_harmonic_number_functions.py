from metrecs.utils import harmonic_number as hn
from metrecs.utils import normalized_scaled_harmonic_number_series as norm_hn_series


def test_harmonic_number():
    # Source: https://en.wikipedia.org/wiki/Harmonic_number
    assert round(hn(1), 2) == 1.00
    assert round(hn(2), 3) == 1.50
    assert round(hn(3), 4) == 1.8333
    assert round(hn(4), 5) == 2.08333
    assert round(hn(5), 5) == 2.28333
    assert round(hn(6), 5) == 2.45
    assert round(hn(7), 5) == 2.59286
    assert round(hn(8), 5) == 2.71786
    assert round(hn(9), 5) == 2.82897
    assert round(hn(10), 5) == 2.92897
    assert round(hn(20), 5) == 3.59774
    assert round(hn(30), 5) == 3.99499
    assert round(hn(40), 5) == 4.27854


def test_normalized_harmonic_number_series():
    assert 1 / hn(1) == norm_hn_series(1)
    assert (
        round(sum(norm_hn_series(5)), 2) == 1.0
    ), "'normalized_harmonic_number_series(n)' output must sum to 1.0"
