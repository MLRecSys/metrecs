import numpy as np
import math
from scipy.special import kl_div


def harmonic_number(n):
    """Returns an approximate value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)

a = np.array(["a", "a", "b", "c"])
b = np.array(["a", "b", "b", "c"])



def histogram(a: np.array, adjusted = False):
    n = np.unique(a, return_counts = True)[1]
    sum_one_over_ranks = harmonic_number(len(a))
    if adjusted:
        p = n * 1 / np.sum(n) / sum_one_over_ranks
    else:
        p = n * 1 / np.sum(n)
    # d = [np.array([i] * j) for i,j in zip(p,n)]
    # d = np.concatenate(d)
    return(p)



da = histogram(a, adjusted = False)
db = histogram(b, adjusted = True)
kl_div(da, db)



pop_recs = np.array([["a", "a", "b", "c"],
                     ["a", "c", "b", "c"]])

own_recs = np.array(["a", "a", "b", "c"])

np.repeat(own_recs, 3)



def fragmentation(pop_recs, own_recs):
    # TODO: for vectorization
    # own_recs_hist_rep = np.tile(own_recs_hist, (pop_recs.shape[0], 1))

    own_recs_hist = histogram(own_recs)
    pop_recs_hist = np.apply_along_axis(histogram, 1, pop_recs)
    div = np.mean([kl_div(own_recs_hist, p) for p in pop_recs_hist])
    return(div)





n = len(a)


distr = {}
distr.get("v", 0.)

distr["v"] = 2 

d = a

r = np.arange(1,len(a) + 1)

d = [1 / len(a)] * len(a)


[p[0]] * d[0]
[p] * d
d
a
d
np.histogram(a)

np.count_nonzero(a)

from collections import Counter

counts = Counter(a)
np.array(list(counts.values()))
np.char.count(a, '*')

         
np.array(counts) / len(a)

from statistics import mode

[mode([x[i] for x in a]) for i in np.arange(len(a[0]))]

for 

for indx, item in enumerate(a):
    rank = indx + 1
    story_freq = distr.get(item, 0.)
    distr[item] = story_freq + 1 / rank / sum_one_over_ranks if adjusted else story_freq + 1 / n
    count += 1
