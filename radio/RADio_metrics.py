from sklearn.preprocessing import KBinsDiscretizer
from utils import harmonic_number
from scipy.special import kl_div
import numpy as np


class Calibration:
    """
    Class that calibrates recommender Calibration.
    Theory: https://dl.acm.org/doi/10.1145/3240323.3240372
    Implementation: http://ethen8181.github.io/machine-learning/recsys/calibration/calibrated_reco.html.
    """

    def __init__(self, config):
        n_bins = 5
        self.bins_discretizer = KBinsDiscretizer(
            encode="ordinal", n_bins=n_bins, strategy="uniform"
        )
        # warnings.filterwarnings("ignore", category=UserWarning)
        self.language = config["language"]
        # self.textstat = dart.handler.other.textstat.TextStatHandler(self.language)

    def compute_distr(self, items, adjusted=False):
        """Compute the genre distribution for a given list of Items."""
        n = len(items)
        sum_one_over_ranks = harmonic_number(n)
        count = 0
        distr = {}
        for _, item in items.iterrows():
            count += 1
            topic_freq = distr.get(item.category, 0.0)
            distr[item.category] = (
                topic_freq + 1 * 1 / count / sum_one_over_ranks
                if adjusted
                else topic_freq + 1 * 1 / n
            )
        return distr

    def compute_distr_complexity(self, arr, bins_discretizer, adjusted=False):
        """
        Args:
        Return"
        """
        n = len(arr)
        sum_one_over_ranks = harmonic_number(n)
        arr_binned = bins_discretizer.transform(arr)
        distr = {}
        if adjusted:
            for bin in list(range(bins_discretizer.n_bins)):
                for indx, ele in enumerate(arr_binned[:, 0]):
                    if ele == bin:
                        rank = indx + 1
                        bin_freq = distr.get(bin, 0.0)
                        distr[bin] = bin_freq + 1 * 1 / rank / sum_one_over_ranks

        else:
            for bin in list(range(bins_discretizer.n_bins)):
                distr[bin] = round(
                    np.count_nonzero(arr_binned == bin) / arr_binned.shape[0], 3
                )
        return distr

    def topic_divergence(self, reading_history, recommendation):
        freq_rec = self.compute_distr(recommendation, adjusted=True)
        freq_history = self.compute_distr(reading_history, adjusted=True)
        divergence_with_discount = kl_div(freq_history, freq_rec)

        freq_rec = self.compute_distr(recommendation, adjusted=False)
        freq_history = self.compute_distr(reading_history, adjusted=False)
        divergence_without_discount = kl_div(freq_history, freq_rec)
        return [divergence_with_discount, divergence_without_discount]

    def complexity_divergence(self, reading_history, recommendation):
        if "complexity" in reading_history.columns:
            reading_history_complexity = np.array(reading_history.complexity).reshape(
                -1, 1
            )
            recommendation_complexity = np.array(recommendation.complexity).reshape(
                -1, 1
            )
        # else:
        #     reading_history_complexity = np.array(reading_history.text.apply(lambda x: self.textstat.flesch_kincaid_score(x))).reshape(-1, 1)
        #     recommendation_complexity = np.array(
        #         recommendation.text.apply(lambda x: self.textstat.flesch_kincaid_score(x))).reshape(-1, 1)

        self.bins_discretizer.fit(reading_history_complexity)

        distr_pool = self.compute_distr_complexity(
            reading_history_complexity, self.bins_discretizer, True
        )
        distr_recommendation = self.compute_distr_complexity(
            recommendation_complexity, self.bins_discretizer, True
        )
        divergence_with_discount = kl_div(distr_pool, distr_recommendation)

        distr_pool = self.compute_distr_complexity(
            reading_history_complexity, self.bins_discretizer, False
        )
        distr_recommendation = self.compute_distr_complexity(
            recommendation_complexity, self.bins_discretizer, False
        )
        divergence_without_discount = kl_div(distr_pool, distr_recommendation)
        return [divergence_with_discount, divergence_without_discount]

    def calculate(self, reading_history, recommendation, complexity=True):
        if not reading_history.empty:
            topic_divergence = self.topic_divergence(reading_history, recommendation)
            if complexity:
                complexity_divergence = self.complexity_divergence(
                    reading_history, recommendation
                )
            else:
                complexity_divergence = 0
            return topic_divergence, complexity_divergence
        else:
            return


class Fragmentation:
    """
    Class that calculates to what extent users have seen the same news stories.
    A "story" is considered a set of articles that are about the same 'event'.
    """

    def __init__(self):
        pass

    def compute_distr(self, items, adjusted=False):
        """Compute the genre distribution for a given list of Items."""
        n = len(items)
        sum_one_over_ranks = harmonic_number(n)
        count = 0
        distr = {}
        for indx, item in enumerate(items):
            rank = indx + 1
            story_freq = distr.get(item, 0.0)
            distr[item] = (
                story_freq + 1 * 1 / rank / sum_one_over_ranks
                if adjusted
                else story_freq + 1 * 1 / n
            )
            count += 1

        return distr

    def calculate(self, sample, recommendation):
        with_discount = []
        without_discount = []
        stories_x = recommendation.story.tolist()
        for y in sample:
            try:
                stories_y = y.story.tolist()
                values = self.compare_recommendations(stories_x, stories_y)
                if values:
                    with_discount.append(values[0])
                    without_discount.append(values[1])
            except AttributeError:
                pass
        means_with_discount = [
            np.mean([f[0] for f in with_discount]),
            np.mean([f[1] for f in with_discount]),
        ]
        means_without_discount = [
            np.mean([f[0] for f in without_discount]),
            np.mean([f[1] for f in without_discount]),
        ]
        return [means_with_discount, means_without_discount]

    def compare_recommendations(self, x, y):
        if x and y:
            # output = rbo(x, y, 0.9)
            freq_x = self.compute_distr(x, adjusted=True)
            freq_y = self.compute_distr(y, adjusted=True)
            divergence_with_discount = compute_kl_divergence(freq_x, freq_y)

            freq_x = self.compute_distr(x, adjusted=False)
            freq_y = self.compute_distr(y, adjusted=False)
            divergence_without_discount = compute_kl_divergence(freq_x, freq_y)
            # xy = compute_kl_divergence(freq_x, freq_y)
            # yx = compute_kl_divergence(freq_y, freq_x)
            # kl = 1/2*(xy[0] + yx[0])
            # jsd = 1/2*(xy[1] + yx[1])
            return [divergence_with_discount, divergence_without_discount]
        else:
            return
