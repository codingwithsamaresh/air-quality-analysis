<<<<<<< HEAD
# air-quality-analysis
=======
# Air Quality Analysis and Next-Day AQI Forecasting

A research-oriented machine learning pipeline for analyzing air pollution patterns across Indian cities and forecasting **next-day Air Quality Index (AQI)** using historical pollutant and temporal information.

The project combines **exploratory data analysis, statistical analysis, unsupervised learning, supervised learning, time-aware forecasting, walk-forward validation, residual analysis, and feature importance**.

---

## Overview

Air pollution varies substantially across cities and over time. Understanding these variations requires more than simply predicting AQI from same-day pollutant concentrations.

This project investigates two related problems:

1. **AQI estimation:** How accurately can AQI be estimated from observed pollutant concentrations?
2. **Next-day AQI forecasting:** Can tomorrow's AQI be predicted using only information that would have been available up to today?

A key focus of the project is **avoiding temporal data leakage**. The forecasting models do not use current-day pollutant values to predict the next day's AQI. Instead, they use historical AQI, lagged pollutant measurements, rolling statistics, temporal trends, and calendar features.

---

## Research Questions

The project investigates the following questions:

* Which pollutants are most strongly associated with AQI?
* Is there significant multicollinearity among pollutant variables?
* Can pollution observations be separated into meaningful pollution regimes?
* Are the identified pollution regimes statistically different?
* How accurately can AQI be estimated from pollutant concentrations?
* How accurately can next-day AQI be forecast using historical information?
* Does machine learning outperform a simple previous-day persistence baseline?
* How stable are forecasting models under chronological walk-forward validation?
* Which historical variables contribute most to next-day AQI prediction?
* Where do the forecasting models make their largest errors?

---

## Dataset

The project uses the **Air Quality Data in India** dataset from Kaggle.

Dataset:

> Air Quality Data in India — `city_day.csv`

The dataset contains daily air-quality measurements for multiple Indian cities.

### Dataset characteristics

* **Observations:** 29,531
* **Columns:** 16
* **Time variable:** `Date`
* **Location variable:** `City`
* **Target variable:** `AQI`

### Pollutants

The analysis uses the following pollutant variables:

* PM2.5
* PM10
* NO2
* NOx
* NH3
* CO
* SO2
* O3

---

## Project Architecture

```text
air-quality-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── city_day.csv
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── clustering.py
│   ├── statistical_tests.py
│   ├── regression.py
│   ├── evaluation.py
│   ├── forecasting.py
│   └── main.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
└── tests/
    ├── test_preprocessing.py
    └── test_models.py
```

---

# Methodology

## 1. Data Loading and Preprocessing

The raw dataset is loaded and the `Date` column is converted to a datetime representation.

Different preprocessing strategies are used depending on the analysis.

### Regression

Rows without a target AQI value are removed. Pollutant missing values are handled through median imputation inside the model pipeline.

### Clustering

Pollutant features are:

1. Median-imputed
2. Standardized using `StandardScaler`
3. Passed to K-Means clustering

### Forecasting

A complete **City × Date** panel is constructed to preserve the temporal structure of each city.

Historical features are then generated separately for each city.

---

# 2. Feature Analysis

Three complementary techniques are used:

### Pearson Correlation

Measures the linear relationship between each pollutant and AQI.

### Variance Inflation Factor (VIF)

Used to identify potential multicollinearity among pollutant variables.

### Mutual Information

Measures nonlinear statistical dependence between pollutants and AQI.

---

## Correlation with AQI

| Pollutant | Correlation |
| --------- | ----------: |
| PM10      |       0.803 |
| CO        |       0.683 |
| PM2.5     |       0.659 |
| NO2       |       0.537 |
| SO2       |       0.491 |
| NOx       |       0.486 |
| NH3       |       0.252 |
| O3        |       0.199 |

PM10 has the strongest linear correlation with AQI, followed by CO and PM2.5.

---

## Mutual Information

| Pollutant | Mutual Information |
| --------- | -----------------: |
| PM2.5     |              0.845 |
| PM10      |              0.666 |
| CO        |              0.340 |
| NOx       |              0.198 |
| NO2       |              0.191 |
| NH3       |              0.168 |
| SO2       |              0.122 |
| O3        |              0.110 |

The mutual-information analysis identifies **PM2.5 and PM10** as the strongest sources of statistical information about AQI.

This complements the correlation analysis by capturing potentially nonlinear relationships.

---

## VIF Analysis

The maximum observed VIF is approximately **2.12** for NO2.

All features have VIF values well below commonly used thresholds such as 5 or 10, suggesting that severe multicollinearity is not present among the selected pollutant variables.

---

# 3. Pollution Regime Discovery

K-Means clustering is used to identify groups of observations with similar pollutant profiles.

The number of clusters is evaluated using:

* Within-cluster inertia
* Silhouette score

### K-Means evaluation

|  K |    Inertia | Silhouette |
| -: | ---------: | ---------: |
|  2 | 154,588.61 |  **0.436** |
|  3 | 131,078.27 |      0.405 |
|  4 | 117,368.82 |      0.233 |
|  5 | 105,267.78 |      0.240 |
|  6 |  97,291.23 |      0.243 |
|  7 |  91,591.55 |      0.237 |

Based on the silhouette score, **K = 2** is selected.

### Cluster interpretation

The two clusters correspond approximately to:

* **High-pollution regime**
* **Lower-pollution regime**

| Pollutant | High Pollution | Lower Pollution |
| --------- | -------------: | --------------: |
| PM2.5     |         139.36 |           50.47 |
| PM10      |         246.61 |           90.79 |
| NO2       |          61.79 |           21.26 |
| NOx       |          73.09 |           22.33 |
| NH3       |          36.12 |           21.60 |
| CO        |           7.14 |            1.22 |
| SO2       |          29.50 |           10.79 |
| O3        |          45.04 |           32.54 |

---

# 4. Statistical Validation of Pollution Regimes

After clustering, Welch's independent-samples t-test is used to compare pollutant distributions between the two clusters.

Because multiple hypothesis tests are performed, **Benjamini-Hochberg False Discovery Rate (FDR)** correction is applied.

All eight pollutants remain statistically significant after correction.

This provides statistical evidence that the two clusters represent genuinely different pollution regimes rather than arbitrary K-Means partitions.

---

# 5. Same-Day AQI Estimation

Three supervised learning models are compared:

* Linear Regression
* Random Forest
* Gradient Boosting

A chronological train/test split is used rather than random shuffling.

### Results

| Model                 |       MAE |      RMSE |        R² |
| --------------------- | --------: | --------: | --------: |
| **Gradient Boosting** |     20.51 | **41.36** | **0.882** |
| Random Forest         | **20.39** |     42.19 |     0.877 |
| Linear Regression     |     25.31 |     44.38 |     0.864 |

### Interpretation

Gradient Boosting achieves the best overall performance in terms of **RMSE and R²**, while Random Forest obtains the lowest MAE.

The results demonstrate that pollutant concentrations contain substantial information about the observed AQI.

---

# 6. Leakage-Safe Next-Day AQI Forecasting

A major objective of the project is to distinguish **AQI estimation** from genuine forecasting.

A model that uses today's pollutant measurements to estimate today's AQI is not forecasting tomorrow's AQI.

Therefore, the forecasting pipeline predicts:

```text
Today's available information
            ↓
      Forecasting model
            ↓
      Tomorrow's AQI
```

Current-day pollutant values are excluded from the forecasting feature set.

---

## Forecasting Features

Historical features include:

### AQI lags

* AQI lag 1
* AQI lag 2
* AQI lag 3
* AQI lag 7

### Historical pollutant lags

For each pollutant:

* 1-day lag
* 2-day lag
* 3-day lag
* 7-day lag

### Rolling statistics

Historical rolling means and standard deviations over:

* 3 days
* 7 days
* 14 days

### Temporal features

* Month
* Day of year
* Cyclic month representation
* Cyclic day-of-year representation

### Trend features

* AQI change over 1 day
* AQI change over 3 days
* AQI change over 7 days
* AQI volatility

---

# 7. Forecasting Models

The following models are compared:

* Naive previous-day AQI
* Linear Regression
* Random Forest
* Gradient Boosting

The naive model provides an important baseline:

```text
Predicted AQI(t+1) = AQI(t)
```

A forecasting model should be evaluated against this persistence baseline rather than against accuracy alone.

---

## Forecasting Results

| Model                  |       MAE |      RMSE |        R² |
| ---------------------- | --------: | --------: | --------: |
| **Linear Regression**  |     27.26 | **56.60** | **0.775** |
| Naive Previous-Day AQI | **26.34** |     59.83 |     0.744 |
| Random Forest          |     31.07 |     60.53 |     0.743 |
| Gradient Boosting      |     30.16 |     62.31 |     0.728 |

### Key observation

Linear Regression outperforms the naive persistence baseline in:

* RMSE
* R²

while the naive model has slightly lower MAE.

This indicates that historical features provide useful predictive information beyond simply assuming that tomorrow's AQI will equal today's AQI.

---

# 8. Walk-Forward Validation

A single train/test split can hide temporal instability.

Therefore, expanding-window **walk-forward validation** is used.

The model is repeatedly trained on historical data and evaluated on a future time window.

```text
Fold 1

Train ──────────────────► Validation
2015 ─────────── 2018     2019 Q1


Fold 2

Train ─────────────────────────► Validation
2015 ───────────────── 2019 Q1   2019 Q2


Fold 3

Train ─────────────────────────────────► Validation
2015 ───────────────────── 2019 Q2      2019 Q3


Fold 4

Train ─────────────────────────────────────────► Validation
2015 ─────────────────────────── 2019 Q3         2019 Q4
```

This provides a more realistic estimate of how forecasting models behave when deployed over time.

---

## Walk-Forward Results

| Model                 |  Mean MAE | Mean RMSE |   Mean R² |
| --------------------- | --------: | --------: | --------: |
| **Linear Regression** | **31.28** | **63.12** | **0.755** |
| Gradient Boosting     |     32.50 |     65.80 |     0.735 |
| Random Forest         |     34.58 |     66.07 |     0.730 |

Linear Regression provides the best average performance across the four temporal validation windows.

The variation between folds also demonstrates that forecasting difficulty changes over time.

---

# 9. Residual Analysis

Residuals are calculated as:

```text
Residual = Actual AQI − Predicted AQI
```

For the selected forecasting model, Linear Regression:

| Statistic       | Value |
| --------------- | ----: |
| Mean residual   | -4.57 |
| Median residual | -5.51 |
| Std. residual   | 56.42 |
| MAE             | 27.26 |
| RMSE            | 56.60 |

Residual analysis is used to investigate:

* systematic bias
* large prediction errors
* outliers
* heteroscedasticity
* temporal error patterns

The project also stores residual-level predictions and errors for further analysis.

---

# 10. Feature Importance

Two complementary approaches are used:

### Model-based feature importance

Tree-based feature importance is extracted from Random Forest and Gradient Boosting.

### Permutation importance

Features are randomly permuted and the resulting degradation in predictive performance is measured.

Permutation importance is particularly useful because it evaluates how much predictive information is lost when a feature is disrupted.

---

## Important Forecasting Features

Across the models, the strongest features consistently include:

* `AQI_lag_1`
* `AQI_rolling_mean_7`
* `AQI_rolling_mean_14`
* `AQI_change_1`
* `AQI_change_3`
* `AQI_change_7`
* `PM2.5_lag_1`
* `PM2.5_rolling_mean_3`
* `CO_lag_1`
* historical NO2 statistics

This suggests that **recent AQI dynamics and short-term pollutant trends are the dominant predictors of next-day AQI**.

---

# 11. Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Machine Learning

* Scikit-learn

### Statistical Analysis

* SciPy
* Statsmodels

### Development

* Jupyter Notebook
* Git
* GitHub

---

# 12. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd air-quality-analysis
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 13. Dataset Setup

Download the `city_day.csv` file from the Kaggle Air Quality Data in India dataset.

Place it at:

```text
data/raw/city_day.csv
```

The project expects the following structure:

```text
data/
└── raw/
    └── city_day.csv
```

---

# 14. Running the Pipeline

Run the complete analysis using:

```bash
python -m src.main
```

The pipeline performs:

```text
Data loading
      ↓
Feature correlation
      ↓
VIF analysis
      ↓
Mutual information
      ↓
K-Means clustering
      ↓
Statistical testing
      ↓
Same-day regression
      ↓
Next-day forecasting
      ↓
Walk-forward validation
      ↓
Residual analysis
      ↓
Feature importance
      ↓
Permutation importance
```

---

# 15. Generated Outputs

Results are automatically saved under:

```text
outputs/
├── figures/
└── tables/
```

Examples include:

```text
outputs/tables/
├── walk_forward_results.csv
├── walk_forward_summary.csv
├── forecasting_residuals.csv
├── Random_Forest_feature_importance.csv
├── Gradient_Boosting_feature_importance.csv
├── Linear_Regression_permutation_importance.csv
├── Random_Forest_permutation_importance.csv
└── Gradient_Boosting_permutation_importance.csv
```

Figures include:

```text
outputs/figures/
├── forecasting_residuals.png
└── forecasting_actual_vs_predicted.png
```

---

# 16. Reproducibility

A fixed random seed is used throughout the machine-learning pipeline:

```python
RANDOM_STATE = 42
```

Chronological splitting and walk-forward validation are used for forecasting experiments to avoid random temporal mixing.

---

# 17. Key Findings

The main findings of the project are:

### Pollution characteristics

* PM10 has the strongest linear association with AQI.
* PM2.5 has the highest mutual information with AQI.
* Severe multicollinearity is not observed among the selected pollutants.

### Pollution regimes

* K-Means identifies two meaningful pollution regimes.
* The high-pollution regime exhibits substantially higher concentrations across all major pollutants.
* All eight pollutants show statistically significant differences between the regimes after FDR correction.

### AQI estimation

* Gradient Boosting achieves an R² of approximately **0.882**.
* Random Forest achieves the lowest MAE at approximately **20.39**.

### Next-day forecasting

* Leakage-safe forecasting achieves an R² of approximately **0.775** on the held-out period.
* Linear Regression outperforms the persistence baseline in RMSE and R².
* Walk-forward validation gives Linear Regression an average R² of approximately **0.755**.

### Forecasting drivers

Recent historical AQI, AQI trends, and PM2.5/CO historical measurements are among the strongest predictors of next-day AQI.

---

# 18. Limitations

Several limitations remain.

### Missing observations

The original dataset contains substantial missingness. Median imputation is used in several modeling pipelines and may not perfectly represent the underlying temporal process.

### AQI availability

The forecasting task depends on historical AQI availability and therefore evaluates a setting where recent AQI observations are known.

### City heterogeneity

Different cities have different emission sources, climate, geography, and monitoring patterns. A single global model may not capture all city-specific dynamics.

### Extreme events

Large residuals indicate that some extreme AQI events remain difficult to predict.

### Model scope

The current project focuses on classical machine-learning methods. More advanced temporal models such as:

* XGBoost
* LightGBM
* XGBoost with lag engineering
* Temporal Convolutional Networks
* LSTM
* Transformer-based time-series models

could be investigated in future work.

---

# 19. Future Work

Potential extensions include:

* City-specific forecasting models
* Hierarchical city + global models
* Strict temporal imputation
* XGBoost/LightGBM comparison
* Hyperparameter optimization
* SHAP-based explainability
* Prediction intervals and uncertainty estimation
* Extreme-AQI event classification
* Seasonal forecasting analysis
* Spatial modeling between neighboring cities
* Multivariate time-series models
* LSTM/Transformer-based forecasting
* Probabilistic AQI forecasting
* Deployment as an interactive dashboard or API

---

# 20. Project Highlights

This project emphasizes **methodological correctness rather than only maximizing predictive accuracy**.

In particular:

* Temporal leakage was explicitly investigated and removed.
* A naive persistence baseline is included.
* Chronological train/test splitting is used.
* Walk-forward validation evaluates temporal robustness.
* Multiple feature-selection techniques are compared.
* Statistical significance is corrected for multiple testing.
* Residual analysis is performed instead of reporting only model scores.
* Feature importance and permutation importance are used for interpretability.

The central distinction is:

```text
Same-day estimation
Pollutants(t) ─────────────► AQI(t)

              versus

Next-day forecasting
History up to t ───────────► AQI(t+1)
```

This distinction is critical when evaluating real-world air-quality forecasting systems.

---

## Author

**Samaresh Koley**

M.Tech Computer Science
Indian Statistical Institute, Kolkata

---

## License

This project is intended for educational, research, and portfolio purposes.
>>>>>>> e8699e2 (Initial commit: air quality analysis and forecasting)
