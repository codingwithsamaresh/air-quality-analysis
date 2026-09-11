import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def prepare_regression_data(df, features, target):
    """
    Prepare regression data.

    Rows with missing target values are removed.
    Missing feature values are NOT imputed here.
    The regression pipeline will handle imputation
    using training data only.
    """

    data = df.dropna(subset=[target]).copy()

    data = data.sort_values("Date")

    X = data[features]
    y = data[target]

    return data, X, y


def prepare_clustering_data(df, features, target):
    """
    Prepare standardized data for K-Means.

    Median imputation is applied to the clustering
    feature matrix, followed by standardization.
    """

    data = df.dropna(subset=[target]).copy()

    X = data[features]

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return data, X_scaled, imputer, scaler