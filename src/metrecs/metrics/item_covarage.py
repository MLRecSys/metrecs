import numpy as np


def coverage_count(items: np.ndarray) -> int:
    """Fraction of unique items recommended given the full catelog

    Args:
        items (np.ndarray): array with items that has been recommended
    Returns:
        int: count of unique items
    """
    return len(set(items))


def coverage_fraction(items: np.ndarray, catelog: np.ndarray) -> float:
    """Fraction of unique items recommended given the full catelog

    Args:
        items (np.ndarray): array with an item subset of the catelog that was recommended
        catelog (np.ndarray): all items that could be recommended.
    Returns:
        float: covarage score
    """
    return len(set(items)) / len(set(catelog))
