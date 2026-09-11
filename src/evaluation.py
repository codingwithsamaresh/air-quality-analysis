import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_regression(
    model_name,
    y_true,
    predictions
):
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate all models.
    """

    results = []
    predictions = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(X_test)

        results.append(
            evaluate_regression(
                name,
                y_test,
                pred
            )
        )

        predictions[name] = pred

    return (
        pd.DataFrame(results)
        .sort_values("RMSE"),
        predictions
    )