import os
import sys
import datetime
import time
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
import logging
import ipdb

if "src" not in sys.path:
    sys.path.append("src")

from metrecs.utils import (
    harmonic_number,
    normalized_scaled_harmonic_number_series,
    compute_normalized_distribution_multiple_categories,
    opt_merge_max_mappings,
    avoid_distribution_misspecification,
    user_level_RADio_multicategorical,
)


def get_classic_calibration(
    pred_preselection_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    behaviors_presel_df: pd.DataFrame,
    cat_column: str,
    top_at: int,
) -> pd.DataFrame:
    df_cat = dataset.get_cat(
        pred_preselection_df,
        articles_df,
        "pred_slice_id",
        cat_column,
        top_at,
        slice_col=True,
    )

    df_cat_history = dataset.get_cat(
        behaviors_presel_df[["index", "user", "behavior_array"]],
        articles_df,
        "behavior_array",
        cat_column,
        top_at,
        slice_col=False,
    )
    df_cat_history = df_cat_history.rename(
        columns={
            "sorted_cat_list": "history_cat_list",
            "sorted_newsid_list": "history_newsid_list",
        }
    )

    df_calibration = df_cat.merge(df_cat_history, on=["index", "user"], how="inner")
    df_calibration["calibration"] = df_calibration.apply(
        lambda row: user_level_RADio_multicategorical(
            row["sorted_cat_list"],
            row["history_cat_list"],
            list(normalized_scaled_harmonic_number_series(len(row["sorted_cat_list"]))),
        ),
        axis=1,
    )
    return df_calibration


def get_classic_representation(
    pred_preselection_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    behaviors_presel_df: pd.DataFrame,
    cat_column: str,
    top_at: int,
) -> pd.DataFrame:
    df_cat = dataset.get_cat(
        pred_preselection_df,
        articles_df,
        "pred_slice_id",
        cat_column,
        top_at,
        slice_col=True,
    )

    df_cat_pool = dataset.get_cat(
        behaviors_presel_df[["index", "user", "pool_array"]],
        articles_df,
        "pool_array",
        cat_column,
        top_at,
        slice_col=False,
    )
    df_cat_pool = df_cat_pool.rename(
        columns={
            "sorted_cat_list": "pool_cat_list",
            "sorted_newsid_list": "pool_newsid_list",
        }
    )

    df_representation = df_cat.merge(df_cat_pool, on=["index", "user"], how="inner")

    df_representation["representation"] = df_representation.apply(
        lambda row: user_level_RADio_multicategorical(
            row["sorted_cat_list"],
            row["pool_cat_list"],
            list(normalized_scaled_harmonic_number_series(len(row["sorted_cat_list"]))),
        ),
        axis=1,
    )

    return df_representation


def get_classic_fragmentation(
    pred_preselection_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    cat_column: str,
    top_at: int,
    sample_size: float = 0.05,
) -> pd.DataFrame:
    df_cat = dataset.get_cat(
        pred_preselection_df,
        articles_df,
        "pred_slice_id",
        cat_column,
        top_at,
        slice_col=True,
    )
    df_cat2 = df_cat.sample(frac=0.01).rename(
        columns={
            "index": "other_index",
            "user": "other_user",
            "sorted_cat_list": "rec_cat_list",
            "sorted_newsid_list": "rec_newsid",
            "algo": "other_algo",
        }
    )
    df_cat["key"] = 0
    df_cat2["key"] = 0

    df_cat_cat = df_cat.merge(df_cat2, on="key", how="outer")
    df_cat_cat = df_cat_cat.drop(["key"], axis=1)
    df_cat_cat = df_cat_cat[df_cat_cat["index"] != df_cat_cat["other_index"]]

    df_cat_cat["fragmentation_detail"] = df_cat_cat.apply(
        lambda row: user_level_RADio_multicategorical(
            row["sorted_cat_list"],
            row["rec_cat_list"],
            list(normalized_scaled_harmonic_number_series(len(row["sorted_cat_list"]))),
            list(normalized_scaled_harmonic_number_series(len(row["rec_cat_list"]))),
        ),
        axis=1,
    )
    df_fragmentation = df_cat_cat.groupby(["index", "user", "algo"]).agg(
        {"fragmentation_detail": "mean"}
    )
    df_fragmentation = df_fragmentation.rename(
        columns={"fragmentation_detail": "fragmentation"}
    ).reset_index()
    return df_fragmentation


def calculate_metrics(
    articles_df: pd.DataFrame,
    behaviors_presel_df: pd.DataFrame,
    pred_preselection_df: pd.DataFrame,
    algo: str,
    top_at: int = 10,
    NR_BINS: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Calculating calibration")
    df_calibration = get_classic_calibration(
        pred_preselection_df,
        articles_df,
        behaviors_presel_df,
        "cat_as_list",
        top_at,
    )

    df_calibration = df_calibration[["index", "user", "calibration", "algo"]]

    df_calibration["calibration_bin"] = df_calibration.apply(
        lambda row: round(row["calibration"] * NR_BINS, 0) / NR_BINS, axis=1
    )

    logger.info("Calculating representation")
    df_representation = get_classic_representation(
        pred_preselection_df, articles_df, behaviors_presel_df, "cat_subcat", top_at
    )
    df_representation = df_representation[["index", "user", "representation", "algo"]]

    df_representation["representation_bin"] = df_representation.apply(
        lambda row: round(row["representation"] * NR_BINS, 0) / NR_BINS, axis=1
    )

    logger.info("Calculating fragmentation")
    df_fragmentation = get_classic_fragmentation(
        pred_preselection_df, articles_df, "cat_subcat", top_at, sample_size=0.02
    )

    if df_fragmentation.shape[0] > 0:
        df_fragmentation["fragmentation_bin"] = df_fragmentation.apply(
            lambda row: round(row["fragmentation"] * NR_BINS, 0) / NR_BINS, axis=1
        )

    return (df_calibration, df_representation, df_fragmentation)


def main(
    path_behaviours: str,
    path_articles: str,
    path_predictions: str,
    path_results: str,
    algo: str,
    top_at: int,
):
    articles_df = dataset.read_articles(path=PATH_ARTICLES)
    behaviors_presel_df = dataset.read_behavior(path=PATH_BEHAVIOURS)
    _, pred_preselection_df = dataset.read_predictions(
        path_predictions, behaviors_presel_df, top_k=top_at, algo=algo
    )
    # pred_preselection_df = pred_preselection_df.sample(2000)
    # behaviors_presel_df = behaviors_presel_df.sample(50000)

    df_calibration, df_representation, df_fragmentation = calculate_metrics(
        articles_df,
        behaviors_presel_df,
        pred_preselection_df,
        algo=algo,
        top_at=top_at,
        NR_BINS=200,
    )

    df_calibration.to_parquet(
        os.path.join(path_results, f"{algo}_at_{top_at}_calibration.parquet")
    )
    df_representation.to_parquet(
        os.path.join(path_results, f"{algo}_at_{top_at}_representation.parquet")
    )
    df_fragmentation.to_parquet(
        os.path.join(path_results, f"{algo}_at_{top_at}_fragmentation.parquet")
    )

    return df_calibration, df_representation, df_fragmentation


if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algo", default="pop")
    parser.add_argument("-d", "--dataset", default="MIND")
    parser.add_argument("-t", "--top_at", default="9")
    args = parser.parse_args()

    ALGO = args.algo
    TOP_ATS = [int(top_at) for top_at in args.top_at.split(",")]
    DATASET = args.dataset

    if DATASET == "Globo":
        from utils import globo as dataset
    else:
        from utils import mind as dataset

    with open("examples/paths.json") as f:
        paths = json.load(f)

    PATH_BEHAVIOURS = paths[DATASET]["behaviours"]
    PATH_ARTICLES = paths[DATASET]["articles"]
    PATH_PREDICTIONS = paths[DATASET]["predictions"][ALGO]
    PATH_RESULTS = f"examples/results/{DATASET}/"

    if not os.path.exists(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)

    for TOP_AT in TOP_ATS:
        logging.basicConfig(
            filename=os.path.join(PATH_RESULTS, f"{ALGO}_at_{TOP_AT}.log"),
            filemode="w",
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y %m %d %H:%M:%S",
            level=logging.INFO,
        )

        logger = logging.getLogger(f"{ALGO}@{TOP_AT}-logger")
        logger.addHandler(logging.StreamHandler(sys.stdout))
        print(f"Running metrics on {ALGO} for top at {TOP_AT}")
        # try:
        main(
            path_behaviours=PATH_BEHAVIOURS,
            path_articles=PATH_ARTICLES,
            path_predictions=PATH_PREDICTIONS,
            path_results=PATH_RESULTS,
            algo=ALGO,
            top_at=TOP_AT,
        )
        logger.info(f"Finished running metrics on {ALGO} for top at {TOP_AT}")
        # except Exception as e:
        #     message = "Error running experiment:" + str(e)
        #     print(message)
        #     # raise ValueError(message)
        #     logger.error(message)

logger.info(f"Finished all metrics for algo {ALGO}")

# python examples/run_radio_metric.py --dataset=Globo --algo=top24h
# python examples/run_radio_metric.py --dataset=MIND --algo=pop --top_at=5
