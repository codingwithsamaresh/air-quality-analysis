import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor


def correlation_with_target(df, features, target):
    """
    Calculate Pearson correlation between each feature
    and the target.
    """

    correlations = (
        df[features + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    return correlations


def calculate_vif(df, features):
    """
    Calculate Variance Inflation Factor (VIF).
    """

    X = df[features]

    imputer = SimpleImputer(strategy="median")

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=features
    )

    vif = pd.DataFrame({
        "feature": features,
        "VIF": [
            variance_inflation_factor(
                X_imputed.values,
                i
            )
            for i in range(len(features))
        ]
    })

    return vif.sort_values(
        "VIF",
        ascending=False
    )


def calculate_mutual_information(
    df,
    features,
    target,
    random_state=42
):
    """
    Estimate nonlinear dependency between features
    and the regression target.
    """

    X = df[features]
    y = df[target]

    imputer = SimpleImputer(strategy="median")

    X_imputed = imputer.fit_transform(X)

    mi = mutual_info_regression(
        X_imputed,
        y,
        random_state=random_state
    )

    result = pd.DataFrame({
        "feature": features,
        "mutual_information": mi
    })

    return result.sort_values(
        "mutual_information",
        ascending=False
    )