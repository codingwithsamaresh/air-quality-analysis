import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def evaluate_k_values(
    X,
    k_range,
    random_state=42
):
    """
    Evaluate K-Means for different values of K.
    """

    results = []

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(
            X,
            labels
        )

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": silhouette
        })

    return pd.DataFrame(results)


def fit_kmeans(
    X,
    n_clusters,
    random_state=42
):
    """
    Fit final K-Means model.
    """

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels = model.fit_predict(X)

    return model, labels


def cluster_profile(
    df,
    features,
    labels
):
    """
    Calculate mean pollutant profile for each cluster.
    """

    data = df.copy()
    data["Cluster"] = labels

    profile = (
        data.groupby("Cluster")[features]
        .mean()
    )

    return profile