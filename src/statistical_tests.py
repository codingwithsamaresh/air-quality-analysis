import pandas as pd

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests


def compare_clusters(
    df,
    features,
    cluster_column="Cluster"
):
    """
    Compare pollutant distributions between
    two clusters using Welch's t-test.
    """

    clusters = sorted(
        df[cluster_column].unique()
    )

    if len(clusters) != 2:
        raise ValueError(
            "compare_clusters currently expects exactly 2 clusters."
        )

    cluster_a, cluster_b = clusters

    results = []

    for feature in features:

        group_a = df.loc[
            df[cluster_column] == cluster_a,
            feature
        ].dropna()

        group_b = df.loc[
            df[cluster_column] == cluster_b,
            feature
        ].dropna()

        statistic, p_value = ttest_ind(
            group_a,
            group_b,
            equal_var=False
        )

        results.append({
            "feature": feature,
            "cluster_0_mean": group_a.mean(),
            "cluster_1_mean": group_b.mean(),
            "t_statistic": statistic,
            "p_value": p_value
        })

    results = pd.DataFrame(results)

    reject, adjusted_p, _, _ = multipletests(
        results["p_value"],
        method="fdr_bh"
    )

    results["adjusted_p_value"] = adjusted_p
    results["significant"] = reject

    return results.sort_values(
        "adjusted_p_value"
    )