from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from .config import FIGURES_DIR

from .config import (
    POLLUTANT_FEATURES,
    TARGET,
    RANDOM_STATE,
    K_RANGE,
    REGRESSION_SPLIT_DATE,
    TABLES_DIR
)

from .forecasting import (
    prepare_forecasting_data,
    time_split_forecasting_data,
    build_forecasting_models,
    evaluate_forecasting_models,
    naive_forecast,
    evaluate_naive_forecast,
    walk_forward_validation,
    summarize_walk_forward_results,
    create_residual_dataframe,
    residual_statistics,
    get_feature_importance,
    get_permutation_importance
)

from .data_loader import load_data

from .preprocessing import (
    prepare_regression_data,
    prepare_clustering_data
)

from .feature_selection import (
    correlation_with_target,
    calculate_vif,
    calculate_mutual_information
)

from .clustering import (
    evaluate_k_values,
    fit_kmeans,
    cluster_profile
)

from .statistical_tests import (
    compare_clusters
)

from .regression import (
    build_models,
    time_split
)

from .evaluation import (
    evaluate_models
)


def main():

    # Create output directory
    TABLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("AIR QUALITY ANALYSIS")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    df = load_data()

    print(f"\nDataset shape: {df.shape}")

    # ---------------------------------------------------------
    # 2. Feature analysis
    # ---------------------------------------------------------

    print("\n--- Feature Correlation with AQI ---")

    df_with_aqi = df.dropna(
        subset=[TARGET]
    )

    correlations = correlation_with_target(
        df_with_aqi,
        POLLUTANT_FEATURES,
        TARGET
    )

    print(correlations)

    correlations.to_csv(
        TABLES_DIR / "aqi_correlations.csv"
    )

    print("\n--- VIF ---")

    vif = calculate_vif(
        df_with_aqi,
        POLLUTANT_FEATURES
    )

    print(vif)

    vif.to_csv(
        TABLES_DIR / "vif.csv",
        index=False
    )

    print("\n--- Mutual Information ---")

    mi = calculate_mutual_information(
        df_with_aqi,
        POLLUTANT_FEATURES,
        TARGET,
        RANDOM_STATE
    )

    print(mi)

    mi.to_csv(
        TABLES_DIR / "mutual_information.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 3. K-Means
    # ---------------------------------------------------------

    print("\n--- K-Means ---")

    cluster_df, X_scaled, _, _ = (
        prepare_clustering_data(
            df,
            POLLUTANT_FEATURES,
            TARGET
        )
    )

    k_results = evaluate_k_values(
        X_scaled,
        K_RANGE,
        RANDOM_STATE
    )

    print(k_results)

    k_results.to_csv(
        TABLES_DIR / "kmeans_k_selection.csv",
        index=False
    )

    best_k = int(
        k_results.loc[
            k_results["silhouette_score"].idxmax(),
            "k"
        ]
    )

    print(f"\nSelected K: {best_k}")

    kmeans, labels = fit_kmeans(
        X_scaled,
        best_k,
        RANDOM_STATE
    )

    cluster_df["Cluster"] = labels

    profile = cluster_profile(
        cluster_df,
        POLLUTANT_FEATURES,
        labels
    )

    print("\nCluster profiles:")
    print(profile)

    profile.to_csv(
        TABLES_DIR / "cluster_profiles.csv"
    )
    plt.figure(figsize=(11, 6))

    profile.T.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.xlabel("Pollutant")
    plt.ylabel("Mean Concentration")
    plt.title("Pollution Profiles Across K-Means Clusters")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "cluster_profiles.png",
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # K-Means visualizations
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_results["k"],
        k_results["inertia"],
        marker="o"
    )

    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("K-Means Elbow Curve")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "elbow_curve.png",
        dpi=300
    )

    plt.close()


    plt.figure(figsize=(8, 5))

    plt.plot(
        k_results["k"],
        k_results["silhouette_score"],
        marker="o"
    )

    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "silhouette_scores.png",
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # 4. Statistical validation
    # ---------------------------------------------------------

    if best_k == 2:

        stats = compare_clusters(
            cluster_df,
            POLLUTANT_FEATURES
        )

        print("\n--- Statistical Cluster Comparison ---")
        print(stats)

        stats.to_csv(
            TABLES_DIR / "cluster_statistical_tests.csv",
            index=False
        )

    else:
        print(
            "\nStatistical two-cluster comparison skipped "
            f"because selected K={best_k}."
        )

    # ---------------------------------------------------------
    # 5. Regression
    # ---------------------------------------------------------

    print("\n--- Regression ---")

    regression_df, _, _ = prepare_regression_data(
        df,
        POLLUTANT_FEATURES,
        TARGET
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = time_split(
        regression_df,
        POLLUTANT_FEATURES,
        TARGET,
        REGRESSION_SPLIT_DATE
    )

    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations : {len(X_test)}")

    models = build_models(
        RANDOM_STATE
    )

    results, predictions = evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    best_predictions = predictions["Gradient Boosting"]

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_test,
        best_predictions,
        alpha=0.35
    )

    min_value = min(
        y_test.min(),
        best_predictions.min()
    )

    max_value = max(
        y_test.max(),
        best_predictions.max()
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value]
    )

    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Gradient Boosting: Actual vs Predicted AQI")

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "predicted_vs_actual.png",
        dpi=300
    )

    plt.close()

    print("\n--- Model Comparison ---")
    print(results)

    results.to_csv(
        TABLES_DIR / "model_comparison.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        results["Model"],
        results["RMSE"]
    )

    plt.ylabel("RMSE")
    plt.title("Regression Model Comparison")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "model_rmse_comparison.png",
        dpi=300
    )

    plt.close()

    # ---------------------------------------------------------
    # 6. Time-aware AQI Forecasting
    # ---------------------------------------------------------

    print("\n--- Time-Aware AQI Forecasting ---")

    forecast_df, X_forecast, y_forecast, forecast_features = (
        prepare_forecasting_data(
            df,
            POLLUTANT_FEATURES,
            TARGET
        )
    )

    (
        X_train_forecast,
        X_test_forecast,
        y_train_forecast,
        y_test_forecast,
        train_forecast,
        test_forecast
    ) = time_split_forecasting_data(
        forecast_df,
        forecast_features,
        "AQI_next",
        REGRESSION_SPLIT_DATE
    )

    print(
        f"Forecasting training observations: "
        f"{len(X_train_forecast)}"
    )

    print(
        f"Forecasting testing observations : "
        f"{len(X_test_forecast)}"
    )

    print(
        f"Forecasting features             : "
        f"{len(forecast_features)}"
    )

    # ---------------------------------------------------------
    # Forecasting models
    # ---------------------------------------------------------

    forecast_models = build_forecasting_models(
        RANDOM_STATE
    )

    forecast_results, forecast_predictions = (
        evaluate_forecasting_models(
            forecast_models,
            X_train_forecast,
            X_test_forecast,
            y_train_forecast,
            y_test_forecast
        )
    )

    # ---------------------------------------------------------
    # Naive baseline
    # ---------------------------------------------------------

    naive_data = naive_forecast(
        forecast_df,
        TARGET
    )

    naive_test = naive_data[
        naive_data["Date"] >= pd.Timestamp(
            REGRESSION_SPLIT_DATE
        )
    ].copy()

    naive_result = evaluate_naive_forecast(
        naive_test,
        TARGET
    )

    forecast_results = pd.concat(
        [
            pd.DataFrame([naive_result]),
            forecast_results
        ],
        ignore_index=True
    )

    forecast_results = forecast_results.sort_values(
        "RMSE"
    )

    print("\n--- Forecasting Model Comparison ---")
    print(forecast_results)

    forecast_results.to_csv(
        TABLES_DIR / "forecasting_comparison.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 7. Walk-Forward Validation
    # ---------------------------------------------------------

    print("\n--- Walk-Forward Validation ---")

    walk_forward_results = walk_forward_validation(
        forecast_df,
        forecast_features,
        target="AQI_next",
        split_date=REGRESSION_SPLIT_DATE,
        n_splits=4,
        test_window_days=90,
        random_state=RANDOM_STATE
    )

    print("\n--- Walk-Forward Results ---")
    print(walk_forward_results)

    walk_forward_summary = summarize_walk_forward_results(
        walk_forward_results
    )

    print("\n--- Walk-Forward Summary ---")
    print(walk_forward_summary)

    walk_forward_results.to_csv(
        TABLES_DIR / "walk_forward_results.csv",
        index=False
    )

    walk_forward_summary.to_csv(
        TABLES_DIR / "walk_forward_summary.csv",
        index=False
    )

    # ---------------------------------------------------------
    # 8. Residual Analysis
    # ---------------------------------------------------------

    print("\n--- Residual Analysis ---")

    # Analyze the best model according to RMSE
    best_forecast_model = (
        forecast_results
        .iloc[0]["Model"]
    )

    if best_forecast_model in forecast_predictions:

        best_predictions = forecast_predictions[
            best_forecast_model
        ]

        residual_stats = residual_statistics(
            y_test_forecast,
            best_predictions
        )

        print(
            f"\nBest forecasting model: "
            f"{best_forecast_model}"
        )

        print("\nResidual statistics:")

        for key, value in residual_stats.items():
            print(
                f"{key}: {value:.4f}"
            )

        residual_df = create_residual_dataframe(
            y_test_forecast,
            best_predictions,
            dates=test_forecast["Date"],
            cities=test_forecast["City"]
        )

        residual_df.to_csv(
            TABLES_DIR / "forecasting_residuals.csv",
            index=False
        )

    # ---------------------------------------------------------
    # 9. Feature Importance
    # ---------------------------------------------------------

    print("\n--- Feature Importance ---")

    for model_name in [
        "Random Forest",
        "Gradient Boosting"
    ]:

        model = forecast_models[model_name]

        importance = get_feature_importance(
            model,
            forecast_features
        )

        print(
            f"\nTop features - {model_name}"
        )

        print(
            importance.head(15)
        )

        filename = (
            model_name
            .lower()
            .replace(" ", "_")
            + "_feature_importance.csv"
        )

        importance.to_csv(
            TABLES_DIR / filename,
            index=False
        )

    # ---------------------------------------------------------
    # 10. Permutation Feature Importance
    # ---------------------------------------------------------

    print("\n--- Permutation Feature Importance ---")

    for model_name in [
        "Linear Regression",
        "Random Forest",
        "Gradient Boosting"
    ]:

        model = forecast_models[model_name]

        permutation_importance_df = (
            get_permutation_importance(
                model,
                X_test_forecast,
                y_test_forecast,
                forecast_features,
                random_state=RANDOM_STATE,
                n_repeats=10
            )
        )

        print(
            f"\nTop permutation features - "
            f"{model_name}"
        )

        print(
            permutation_importance_df.head(15)
        )

        filename = (
            model_name
            .lower()
            .replace(" ", "_")
            + "_permutation_importance.csv"
        )

        permutation_importance_df.to_csv(
            TABLES_DIR / filename,
            index=False
        )

    # ---------------------------------------------------------
    # 11. Residual Plots
    # ---------------------------------------------------------

    if best_forecast_model in forecast_predictions:

        residuals = (
            y_test_forecast.values
            - best_predictions
        )

        # Residual vs predicted
        plt.figure(figsize=(8, 6))

        plt.scatter(
            best_predictions,
            residuals,
            alpha=0.4
        )

        plt.axhline(
            0,
            linestyle="--"
        )

        plt.xlabel("Predicted AQI")
        plt.ylabel("Residual")
        plt.title(
            f"Residuals vs Predicted AQI - "
            f"{best_forecast_model}"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "forecasting_residuals.png",
            dpi=300
        )

        plt.close()

        # Actual vs predicted
        plt.figure(figsize=(8, 6))

        plt.scatter(
            y_test_forecast,
            best_predictions,
            alpha=0.4
        )

        min_value = min(
            y_test_forecast.min(),
            best_predictions.min()
        )

        max_value = max(
            y_test_forecast.max(),
            best_predictions.max()
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.title(
            f"Actual vs Predicted AQI - "
            f"{best_forecast_model}"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR
            / "forecasting_actual_vs_predicted.png",
            dpi=300
        )

        plt.close()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()