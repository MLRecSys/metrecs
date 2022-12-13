from metrecs.metrics.item_covarage import coverage_count, coverage_fraction

items = list(range(10)) * 2
catelog = list(range(25))


def test_coverage_count():
    assert coverage_count(items) == 10, "Covarage count gone wrong"


def test_coverage_fraction():
    assert coverage_fraction(items, catelog) == 10 / 25, "Covarage fraction gone wrong"
