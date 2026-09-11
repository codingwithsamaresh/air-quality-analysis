# src/forecasting.py

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# 1. CREATE COMPLETE DAILY CITY PANEL
# =========================================================

def create_daily_panel(df):
    """
    Create a complete City x Date panel.

    The original dataset contains missing calendar dates
    for several cities. Reindexing makes the forecasting
    problem explicitly calendar-day based.

    Missing observations are kept as NaN and are handled
    later by the forecasting feature construction.

    Returns
    -------
    pandas.DataFrame
        Complete city-date panel.
    """

    data = df.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values(
        ["City", "Date"]
    )

    # Keep one observation per City-Date.
    # If duplicates exist, retain the first observation.
    data = data.drop_duplicates(
        subset=["City", "Date"],
        keep="first"
    )

    cities = data["City"].dropna().unique()

    min_date = data["Date"].min()
    max_date = data["Date"].max()

    all_dates = pd.date_range(
        start=min_date,
        end=max_date,
        freq="D"
    )

    complete_index = pd.MultiIndex.from_product(
        [cities, all_dates],
        names=["City", "Date"]
    )

    data = (
        data
        .set_index(["City", "Date"])
        .reindex(complete_index)
        .reset_index()
    )

    return data


# =========================================================
# 2. CREATE FORECASTING FEATURES
# =========================================================

def create_forecasting_features(
    df,
    pollutant_features,
    target="AQI",
    lags=(1, 2, 3, 7),
    rolling_windows=(3, 7, 14)
):
    """
    Create leakage-safe next-day AQI forecasting features.

    Prediction task:

        Information available up to day t
                    ↓
              predict AQI(t+1)

    Features include:

        - Lagged AQI
        - Lagged pollutant concentrations
        - Rolling historical averages
        - AQI changes
        - AQI rolling standard deviation
        - Calendar/seasonal features

    No current-day pollutant concentration is directly
    supplied as a predictor.

    Returns
    -------
    pandas.DataFrame
        Forecasting dataset.
    """

    data = create_daily_panel(df)

    data = data.sort_values(
        ["City", "Date"]
    ).reset_index(drop=True)

    grouped = data.groupby("City")

    # -----------------------------------------------------
    # Target: next calendar day's AQI
    # -----------------------------------------------------

    data["AQI_next"] = (
        grouped[target].shift(-1)
    )

    # -----------------------------------------------------
    # Historical AQI lags
    # -----------------------------------------------------

    for lag in lags:

        data[f"{target}_lag_{lag}"] = (
            grouped[target].shift(lag)
        )

    # -----------------------------------------------------
    # Historical pollutant lags
    # -----------------------------------------------------

    for feature in pollutant_features:

        for lag in lags:

            data[f"{feature}_lag_{lag}"] = (
                grouped[feature].shift(lag)
            )

    # -----------------------------------------------------
    # Historical rolling AQI statistics
    #
    # shift(1) is CRITICAL.
    #
    # It ensures that day t itself is not included when
    # constructing the features used to predict t+1.
    # -----------------------------------------------------

    for window in rolling_windows:

        data[
            f"{target}_rolling_mean_{window}"
        ] = (
            grouped[target]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    window=window,
                    min_periods=1
                )
                .mean()
            )
        )

        data[
            f"{target}_rolling_std_{window}"
        ] = (
            grouped[target]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(
                    window=window,
                    min_periods=2
                )
                .std()
            )
        )

    # -----------------------------------------------------
    # Historical rolling pollutant statistics
    # -----------------------------------------------------

    for feature in pollutant_features:

        for window in rolling_windows:

            data[
                f"{feature}_rolling_mean_{window}"
            ] = (
                grouped[feature]
                .transform(
                    lambda x:
                    x.shift(1)
                    .rolling(
                        window=window,
                        min_periods=1
                    )
                    .mean()
                )
            )

    # -----------------------------------------------------
    # AQI trend / momentum features
    # -----------------------------------------------------

    data["AQI_change_1"] = (
        data[target]
        - data["AQI_lag_1"]
    )

    data["AQI_change_3"] = (
        data[target]
        - data["AQI_lag_3"]
    )

    data["AQI_change_7"] = (
        data[target]
        - data["AQI_lag_7"]
    )

    # -----------------------------------------------------
    # Recent AQI volatility
    # -----------------------------------------------------

    data["AQI_volatility_7"] = (
        grouped[target]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                window=7,
                min_periods=2
            )
            .std()
        )
    )

    # -----------------------------------------------------
    # Calendar features
    # -----------------------------------------------------

    data["month"] = data["Date"].dt.month

    data["day_of_week"] = (
        data["Date"].dt.dayofweek
    )

    data["day_of_year"] = (
        data["Date"].dt.dayofyear
    )

    # Weekend indicator
    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # -----------------------------------------------------
    # Cyclic seasonal encoding
    # -----------------------------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["day_of_year_sin"] = np.sin(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    data["day_of_year_cos"] = np.cos(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    # -----------------------------------------------------
    # Remove rows where target is unavailable
    # -----------------------------------------------------

    data = data.dropna(
        subset=["AQI_next"]
    ).copy()

    return data


# =========================================================
# 3. PREPARE FORECASTING DATA
# =========================================================

def prepare_forecasting_data(
    df,
    pollutant_features,
    target="AQI"
):
    """
    Prepare the forecasting dataset.

    Only lagged, rolling, trend and calendar features
    are returned.

    Current-day pollutant values are deliberately excluded.

    Returns
    -------
    data
    X
    y
    feature_columns
    """

    data = create_forecasting_features(
        df=df,
        pollutant_features=pollutant_features,
        target=target
    )

    # -----------------------------------------------------
    # Explicit feature selection
    # -----------------------------------------------------

    feature_columns = []

    for column in data.columns:

        if "_lag_" in column:
            feature_columns.append(column)

        elif "_rolling_mean_" in column:
            feature_columns.append(column)

        elif "_rolling_std_" in column:
            feature_columns.append(column)

    # Trend features
    feature_columns.extend([
        "AQI_change_1",
        "AQI_change_3",
        "AQI_change_7",
        "AQI_volatility_7"
    ])

    # Calendar features
    feature_columns.extend([
        "month",
        "day_of_week",
        "day_of_year",
        "is_weekend",
        "month_sin",
        "month_cos",
        "day_of_year_sin",
        "day_of_year_cos"
    ])

    X = data[feature_columns]

    y = data["AQI_next"]

    return (
        data,
        X,
        y,
        feature_columns
    )


# =========================================================
# 4. CHRONOLOGICAL TRAIN / TEST SPLIT
# =========================================================

def time_split_forecasting_data(
    data,
    feature_columns,
    target="AQI_next",
    split_date="2019-01-01"
):
    """
    Perform a chronological train/test split.

    Training:
        Date < split_date

    Testing:
        Date >= split_date

    No random shuffling is performed.
    """

    split_date = pd.Timestamp(
        split_date
    )

    data = data.sort_values(
        "Date"
    ).copy()

    train = data[
        data["Date"] < split_date
    ].copy()

    test = data[
        data["Date"] >= split_date
    ].copy()

    X_train = train[feature_columns]

    y_train = train[target]

    X_test = test[feature_columns]

    y_test = test[target]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train,
        test
    )


# =========================================================
# 5. BUILD FORECASTING MODELS
# =========================================================

def build_forecasting_models(
    random_state=42
):
    """
    Build forecasting models.

    Imputation and scaling are performed inside pipelines
    to prevent preprocessing leakage from the test set.
    """

    models = {

        "Linear Regression": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            )
        ]),

        "Random Forest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=random_state,
                    n_jobs=-1,
                    max_features="sqrt"
                )
            )
        ]),

        "Gradient Boosting": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=random_state
                )
            )
        ])
    }

    return models


# =========================================================
# 6. EVALUATION
# =========================================================

def evaluate_forecasting(
    model_name,
    y_true,
    predictions
):
    """
    Calculate forecasting metrics.
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


# =========================================================
# 7. TRAIN AND EVALUATE MODELS
# =========================================================

def evaluate_forecasting_models(
    models,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate all forecasting models.

    Returns
    -------
    results : pandas.DataFrame
    predictions : dict
    """

    results = []

    predictions = {}

    for name, model in models.items():

        print(
            f"Training forecasting model: {name}"
        )

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        metrics = evaluate_forecasting(
            name,
            y_test,
            pred
        )

        results.append(
            metrics
        )

        predictions[name] = pred

    results = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    return (
        results,
        predictions
    )


# =========================================================
# 8. NAIVE PERSISTENCE BASELINE
# =========================================================

def naive_forecast(
    data,
    target="AQI"
):
    """
    Persistence baseline.

    Prediction:

        AQI(t+1) = AQI(t)

    The current day's AQI is used to predict the next
    calendar day's AQI.

    Because the daily panel has already been constructed,
    missing calendar dates are represented explicitly.
    """

    data = data.sort_values(
        ["City", "Date"]
    ).copy()

    data["Naive_Prediction"] = (
        data.groupby("City")[target]
        .shift(0)
    )

    data["Actual_Next_AQI"] = (
        data.groupby("City")[target]
        .shift(-1)
    )

    return data


# =========================================================
# 9. EVALUATE NAIVE BASELINE
# =========================================================

def evaluate_naive_forecast(
    test_data,
    target="AQI"
):
    """
    Evaluate the persistence baseline.
    """

    valid = test_data.dropna(
        subset=[
            "Actual_Next_AQI",
            "Naive_Prediction"
        ]
    )

    mae = mean_absolute_error(
        valid["Actual_Next_AQI"],
        valid["Naive_Prediction"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            valid["Actual_Next_AQI"],
            valid["Naive_Prediction"]
        )
    )

    r2 = r2_score(
        valid["Actual_Next_AQI"],
        valid["Naive_Prediction"]
    )

    return {
        "Model": "Naive Previous-Day AQI",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# 10. FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(
    model,
    feature_columns
):
    """
    Extract feature importance from tree-based models.

    Works with:
        Random Forest
        Gradient Boosting

    Returns
    -------
    pandas.DataFrame
    """

    if not hasattr(
        model,
        "named_steps"
    ):
        raise ValueError(
            "Expected a sklearn Pipeline."
        )

    estimator = model.named_steps[
        "model"
    ]

    if not hasattr(
        estimator,
        "feature_importances_"
    ):
        raise ValueError(
            "Model does not provide feature importances."
        )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": (
            estimator.feature_importances_
        )
    })

    return importance.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

# =========================================================
# 11. WALK-FORWARD VALIDATION
# =========================================================

def walk_forward_validation(
    data,
    feature_columns,
    target="AQI_next",
    split_date="2019-01-01",
    n_splits=4,
    test_window_days=90,
    random_state=42
):
    """
    Perform expanding-window walk-forward validation.

    Example:

        Fold 1:
        TRAIN | TEST

        Fold 2:
        TRAIN -------- | TEST

        Fold 3:
        TRAIN ---------------- | TEST

        Fold 4:
        TRAIN ---------------------- | TEST

    The training set always contains observations strictly
    before the validation period.

    Parameters
    ----------
    data : DataFrame
        Forecasting dataset.

    feature_columns : list
        Forecasting feature names.

    target : str
        Forecast target.

    split_date : str
        Beginning of the walk-forward evaluation period.

    n_splits : int
        Number of chronological validation windows.

    test_window_days : int
        Number of calendar days in each validation window.

    random_state : int
        Random seed.

    Returns
    -------
    DataFrame
        Performance of every model on every fold.
    """

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    initial_train = data[
        data["Date"] < pd.Timestamp(split_date)
    ].copy()

    if len(initial_train) == 0:
        raise ValueError(
            "No training observations before split_date."
        )

    models_template = build_forecasting_models(
        random_state=random_state
    )

    results = []

    validation_start = pd.Timestamp(split_date)

    for fold in range(1, n_splits + 1):

        validation_end = (
            validation_start
            + pd.Timedelta(days=test_window_days)
        )

        train = data[
            data["Date"] < validation_start
        ].copy()

        validation = data[
            (data["Date"] >= validation_start)
            & (data["Date"] < validation_end)
        ].copy()

        if len(validation) == 0:
            break

        X_train = train[feature_columns]
        y_train = train[target]

        X_validation = validation[feature_columns]
        y_validation = validation[target]

        print(
            f"\nWalk-forward fold {fold}"
        )

        print(
            f"Training period: "
            f"{train['Date'].min().date()} "
            f"to "
            f"{train['Date'].max().date()}"
        )

        print(
            f"Validation period: "
            f"{validation['Date'].min().date()} "
            f"to "
            f"{validation['Date'].max().date()}"
        )

        for name, model_template in models_template.items():

            # Fresh model for every fold
            model = build_forecasting_models(
                random_state=random_state
            )[name]

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_validation
            )

            metrics = evaluate_forecasting(
                name,
                y_validation,
                predictions
            )

            metrics["Fold"] = fold
            metrics["Train_Size"] = len(train)
            metrics["Validation_Size"] = len(validation)
            metrics["Validation_Start"] = (
                validation_start
            )
            metrics["Validation_End"] = (
                validation_end
            )

            results.append(metrics)

        validation_start = validation_end

    results = pd.DataFrame(results)

    if results.empty:
        raise ValueError(
            "Walk-forward validation produced no folds."
        )

    return results


# =========================================================
# 12. SUMMARIZE WALK-FORWARD RESULTS
# =========================================================

def summarize_walk_forward_results(
    walk_forward_results
):
    """
    Calculate mean and standard deviation of the
    walk-forward performance across folds.
    """

    summary = (
        walk_forward_results
        .groupby("Model")
        .agg(
            Mean_MAE=("MAE", "mean"),
            Std_MAE=("MAE", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Std_RMSE=("RMSE", "std"),
            Mean_R2=("R2", "mean"),
            Std_R2=("R2", "std")
        )
        .reset_index()
        .sort_values("Mean_RMSE")
    )

    return summary


# =========================================================
# 13. RESIDUAL ANALYSIS
# =========================================================

def calculate_residuals(
    y_true,
    predictions
):
    """
    Calculate prediction residuals.

    Residual:

        residual = actual - predicted
    """

    residuals = (
        np.asarray(y_true)
        - np.asarray(predictions)
    )

    return residuals


def residual_statistics(
    y_true,
    predictions
):
    """
    Calculate descriptive statistics of residuals.
    """

    residuals = calculate_residuals(
        y_true,
        predictions
    )

    statistics = {
        "Mean Residual": np.mean(residuals),
        "Median Residual": np.median(residuals),
        "Std Residual": np.std(residuals),
        "Min Residual": np.min(residuals),
        "Max Residual": np.max(residuals),
        "MAE": mean_absolute_error(
            y_true,
            predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                predictions
            )
        )
    }

    return statistics


def create_residual_dataframe(
    y_true,
    predictions,
    dates=None,
    cities=None
):
    """
    Create a DataFrame containing:

        Actual
        Predicted
        Residual
        Absolute Error
        Date
        City
    """

    residual_df = pd.DataFrame({
        "Actual": np.asarray(y_true),
        "Predicted": np.asarray(predictions)
    })

    residual_df["Residual"] = (
        residual_df["Actual"]
        - residual_df["Predicted"]
    )

    residual_df["Absolute_Error"] = (
        residual_df["Residual"].abs()
    )

    if dates is not None:
        residual_df["Date"] = (
            pd.to_datetime(dates)
            .reset_index(drop=True)
        )

    if cities is not None:
        residual_df["City"] = (
            pd.Series(cities)
            .reset_index(drop=True)
        )

    return residual_df


# =========================================================
# 14. RESIDUAL ANALYSIS PLOT DATA
# =========================================================

def residual_analysis(
    y_true,
    predictions
):
    """
    Return useful data for residual analysis.

    This function does not create plots itself. It returns
    the arrays required for plotting in main.py or a notebook.
    """

    residuals = calculate_residuals(
        y_true,
        predictions
    )

    return {
        "actual": np.asarray(y_true),
        "predicted": np.asarray(predictions),
        "residuals": residuals,
        "absolute_errors": np.abs(residuals)
    }


# =========================================================
# 15. FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(
    model,
    feature_columns
):
    """
    Extract impurity-based feature importance from a
    fitted Random Forest or Gradient Boosting pipeline.
    """

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Expected a sklearn Pipeline."
        )

    estimator = model.named_steps["model"]

    if not hasattr(
        estimator,
        "feature_importances_"
    ):
        raise ValueError(
            "This model does not provide "
            "feature_importances_."
        )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": (
            estimator.feature_importances_
        )
    })

    return (
        importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


# =========================================================
# 16. PERMUTATION FEATURE IMPORTANCE
# =========================================================

def get_permutation_importance(
    model,
    X_test,
    y_test,
    feature_columns,
    random_state=42,
    n_repeats=10
):
    """
    Calculate permutation feature importance on the
    held-out test set.

    Unlike tree impurity importance, permutation importance
    measures how much model performance deteriorates when
    a feature is randomly shuffled.

    This is generally easier to interpret as a measure of
    predictive contribution.
    """

    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": (
            result.importances_mean
        ),
        "importance_std": (
            result.importances_std
        )
    })

    return (
        importance
        .sort_values(
            "importance_mean",
            ascending=False
        )
        .reset_index(drop=True)
    )