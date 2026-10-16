import math
from abc import ABC, abstractmethod
from statistics import mean, median, variance, stdev

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DISTRICT POPULATION CLASS
# ============================================================

class DistrictPopulation:
    def __init__(self, name, years, populations):
        self.name = name
        self.years = np.asarray(years)
        self.populations = np.asarray(populations, dtype=float)

        if len(self.years) != len(self.populations):
            raise ValueError(
                f"{name}: years and populations must have equal lengths."
            )

        if np.any(self.populations < 0):
            raise ValueError(
                f"{name}: population cannot contain negative values."
            )

    def __repr__(self):
        return (
            f"DistrictPopulation("
            f"name='{self.name}', "
            f"years={self.years[0]}-{self.years[-1]}, "
            f"observations={len(self)})"
        )

    def __len__(self):
        return len(self.years)


# ============================================================
# DATA
# Population is in thousands
# ============================================================

years = np.arange(2015, 2025)

districts = [
    DistrictPopulation(
        "Kampala",
        years,
        [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800]
    ),

    DistrictPopulation(
        "Wakiso",
        years,
        [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670]
    ),

    DistrictPopulation(
        "Gulu",
        years,
        [320, 330, 345, 360, 375, 390, 410, 430, 455, 480]
    ),

    # Additional district 1
    DistrictPopulation(
        "Mbarara",
        years,
        [450, 470, 495, 520, 550, 580, 610, 645, 680, 720]
    ),

    # Additional district 2
    DistrictPopulation(
        "Mbale",
        years,
        [420, 435, 450, 470, 495, 520, 545, 575, 605, 640]
    )
]


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

def show_statistics(district):
    data = district.populations

    print(f"\n{district.name}")
    print("-" * 60)

    # statistics module
    print("statistics module:")
    print(f"Mean:     {mean(data):.2f}")
    print(f"Median:   {median(data):.2f}")
    print(f"Variance: {variance(data):.2f}")
    print(f"Std dev:  {stdev(data):.2f}")

    # NumPy
    print("\nNumPy:")
    print(f"Mean:     {np.mean(data):.2f}")
    print(f"Median:   {np.median(data):.2f}")
    print(f"Variance: {np.var(data):.2f}")
    print(f"Std dev:  {np.std(data):.2f}")

    # To make the NumPy variance equivalent to statistics.variance:
    print("\nNumPy with ddof=1:")
    print(f"Variance: {np.var(data, ddof=1):.2f}")
    print(f"Std dev:  {np.std(data, ddof=1):.2f}")


print("\n" + "=" * 70)
print("TASK 2: DESCRIPTIVE STATISTICS")
print("=" * 70)

for district in districts:
    show_statistics(district)


# ============================================================
# 3. YEAR-ON-YEAR GROWTH AND CAGR
# ============================================================

def yoy_growth(population):
    """
    Returns year-on-year growth rates as percentages.
    """
    return (population[1:] / population[:-1] - 1) * 100


def cagr(population):
    """
    Compound Annual Growth Rate.
    """
    number_of_periods = len(population) - 1

    return (
        (population[-1] / population[0])
        ** (1 / number_of_periods)
        - 1
    )


print("\n" + "=" * 70)
print("TASK 3: GROWTH RATES")
print("=" * 70)

cagr_values = {}

for district in districts:
    growth = yoy_growth(district.populations)
    district_cagr = cagr(district.populations)

    cagr_values[district.name] = district_cagr

    print(f"\n{district.name}")
    print("Year-on-year growth (%):")
    print(np.round(growth, 2))
    print(f"CAGR: {district_cagr * 100:.2f}%")


fastest_district = max(
    cagr_values,
    key=cagr_values.get
)

print("\nFastest-growing district in relative terms:")
print(
    f"{fastest_district}: "
    f"{cagr_values[fastest_district] * 100:.2f}% CAGR"
)


# ============================================================
# 4. ABSTRACT FORECASTER
# ============================================================

class Forecaster(ABC):

    @abstractmethod
    def fit(self, years, populations):
        pass

    @abstractmethod
    def predict(self, horizon):
        pass


# ============================================================
# MODEL A: LINEAR TREND
# ============================================================

class LinearTrendForecaster(Forecaster):

    def fit(self, years, populations):
        self.last_year = years[-1]

        # y = mx + c
        self.coefficients = np.polyfit(
            years,
            populations,
            1
        )

        return self

    def predict(self, horizon):
        future_years = np.arange(
            self.last_year + 1,
            self.last_year + horizon + 1
        )

        return np.polyval(
            self.coefficients,
            future_years
        )

    def fitted_values(self, years):
        return np.polyval(
            self.coefficients,
            years
        )


# ============================================================
# MODEL B: EXPONENTIAL / CAGR
# ============================================================

class CAGRForecaster(Forecaster):

    def fit(self, years, populations):
        self.last_year = years[-1]
        self.last_population = populations[-1]

        periods = len(populations) - 1

        self.growth_rate = (
            populations[-1] / populations[0]
        ) ** (1 / periods) - 1

        return self

    def predict(self, horizon):
        predictions = []

        for i in range(1, horizon + 1):
            prediction = (
                self.last_population
                * (1 + self.growth_rate) ** i
            )

            predictions.append(prediction)

        return np.array(predictions)

    def fitted_values(self, years):
        periods = years - years[0]

        return (
            self.last_population
            / (1 + self.growth_rate)
            ** (years[-1] - years[0])
            * (1 + self.growth_rate) ** periods
        )


# ============================================================
# MODEL C: FIBONACCI-RATIO MODEL
# ============================================================

class FibonacciRatioForecaster(Forecaster):

    def fit(self, years, populations):
        self.last_year = years[-1]
        self.last_population = populations[-1]

        return self

    def fibonacci_ratios(self, horizon):
        fibonacci = [1, 1]

        while len(fibonacci) < horizon + 2:
            fibonacci.append(
                fibonacci[-1] + fibonacci[-2]
            )

        ratios = []

        for i in range(horizon):
            ratios.append(
                fibonacci[i + 1] / fibonacci[i]
            )

        return np.array(ratios)

    def predict(self, horizon):
        ratios = self.fibonacci_ratios(horizon)

        predictions = []
        current = self.last_population

        for ratio in ratios:
            current *= ratio
            predictions.append(current)

        return np.array(predictions)

    def fitted_values(self, years):
        # The Fibonacci model does not naturally provide
        # historical fitted values, so return NaN values.
        return np.full(len(years), np.nan)


# ============================================================
# 5. ERROR METRICS
# ============================================================

def MAE(actual, predicted):
    return np.mean(
        np.abs(actual - predicted)
    )


def RMSE(actual, predicted):
    return np.sqrt(
        np.mean((actual - predicted) ** 2)
    )


def MAPE(actual, predicted):
    return np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100


# ============================================================
# VALIDATION
# Train: 2015-2021
# Test: 2022-2024
# ============================================================

train_mask = years <= 2021
test_mask = years >= 2022

train_years = years[train_mask]
test_years = years[test_mask]

model_classes = [
    LinearTrendForecaster,
    CAGRForecaster,
    FibonacciRatioForecaster
]


validation_results = {}
selected_models = {}


print("\n" + "=" * 70)
print("TASK 5: MODEL VALIDATION")
print("=" * 70)


for district in districts:

    train_population = district.populations[train_mask]
    test_population = district.populations[test_mask]

    district_results = []

    print(f"\n{district.name}")
    print("-" * 70)

    print(
        f"{'Model':<30}"
        f"{'MAE':>12}"
        f"{'RMSE':>12}"
        f"{'MAPE (%)':>12}"
    )

    for ModelClass in model_classes:

        model = ModelClass()

        model.fit(
            train_years,
            train_population
        )

        predictions = model.predict(
            len(test_years)
        )

        model_mae = MAE(
            test_population,
            predictions
        )

        model_rmse = RMSE(
            test_population,
            predictions
        )

        model_mape = MAPE(
            test_population,
            predictions
        )

        result = {
            "model": ModelClass.__name__,
            "object": model,
            "predictions": predictions,
            "MAE": model_mae,
            "RMSE": model_rmse,
            "MAPE": model_mape
        }

        district_results.append(result)

        print(
            f"{ModelClass.__name__:<30}"
            f"{model_mae:>12.2f}"
            f"{model_rmse:>12.2f}"
            f"{model_mape:>12.2f}"
        )

    validation_results[district.name] = district_results

    # Select the model with the lowest RMSE
    best = min(
        district_results,
        key=lambda x: x["RMSE"]
    )

    selected_models[district.name] = best["model"]

    print(
        f"\nBest model for {district.name}: "
        f"{best['model']}"
    )


# ============================================================
# 6. FORECAST 2025-2029
# ============================================================

future_years = np.arange(2025, 2030)

forecast_results = {}


print("\n" + "=" * 70)
print("TASK 6: 2025-2029 FORECAST")
print("=" * 70)


for district in districts:

    selected_model_name = selected_models[
        district.name
    ]

    # Find corresponding class
    selected_class = next(
        cls for cls in model_classes
        if cls.__name__ == selected_model_name
    )

    # Refit selected model using ALL historical data
    model = selected_class()

    model.fit(
        district.years,
        district.populations
    )

    forecast = model.predict(5)

    forecast_results[district.name] = {
        "model": model,
        "forecast": forecast
    }

    print(f"\n{district.name}")
    print(f"Selected model: {selected_model_name}")

    for year, value in zip(
        future_years,
        forecast
    ):
        print(
            f"{year}: {value:.2f} thousand"
        )


# ============================================================
# VARIANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL VS FORECAST VARIANCE")
print("=" * 70)

for district in districts:

    actual_variance = np.var(
        district.populations
    )

    forecast = forecast_results[
        district.name
    ]["forecast"]

    forecast_variance = np.var(
        forecast
    )

    print(f"\n{district.name}")
    print(
        f"Actual variance:   "
        f"{actual_variance:.2f}"
    )

    print(
        f"Forecast variance: "
        f"{forecast_variance:.2f}"
    )

    if forecast_variance < actual_variance:
        print(
            "Interpretation: forecast values are "
            "less variable than the historical series."
        )

    elif forecast_variance > actual_variance:
        print(
            "Interpretation: forecast values are "
            "more variable than the historical series."
        )

    else:
        print(
            "Interpretation: forecast variance is "
            "approximately equal to historical variance."
        )


# ============================================================
# 7. PLOT ALL DISTRICTS
# ============================================================

fig, axes = plt.subplots(
    3,
    2,
    figsize=(15, 16)
)

axes = axes.flatten()


for i, district in enumerate(districts):

    ax = axes[i]

    selected_model = forecast_results[
        district.name
    ]["model"]

    forecast = forecast_results[
        district.name
    ]["forecast"]

    # Actual
    ax.plot(
        district.years,
        district.populations,
        marker="o",
        color="black",
        label="Actual"
    )

    # Fitted historical values
    if not isinstance(
        selected_model,
        FibonacciRatioForecaster
    ):

        if hasattr(
            selected_model,
            "fitted_values"
        ):
            fitted = selected_model.fitted_values(
                district.years
            )

            ax.plot(
                district.years,
                fitted,
                linestyle="--",
                color="blue",
                label="Fitted"
            )

    # Forecast
    ax.plot(
        future_years,
        forecast,
        marker="o",
        linestyle="--",
        color="red",
        label="Forecast"
    )

    # Train/test boundary
    ax.axvline(
        x=2021.5,
        color="green",
        linestyle=":",
        linewidth=2,
        label="Train/Test split"
    )

    ax.set_title(
        f"{district.name} Population"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Population (thousands)")

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()


# Hide unused subplot
for j in range(len(districts), len(axes)):
    axes[j].set_visible(False)


fig.suptitle(
    "UBOS District Population Forecasts",
    fontsize=18
)

plt.tight_layout()
plt.show()


# ============================================================
# 8. CLASSROOM PLANNING
# ============================================================

PRIMARY_SCHOOL_SHARE = 0.18
CLASSROOM_CAPACITY = 53


print("\n" + "=" * 70)
print("TASK 8: CLASSROOM PLANNING")
print("=" * 70)


for district in districts:

    # 2024 actual population
    population_2024 = (
        district.populations[-1] * 1000
    )

    # 2029 forecast
    population_2029 = (
        forecast_results[
            district.name
        ]["forecast"][-1] * 1000
    )

    # Primary-school-age population
    pupils_2024 = (
        population_2024
        * PRIMARY_SCHOOL_SHARE
    )

    pupils_2029 = (
        population_2029
        * PRIMARY_SCHOOL_SHARE
    )

    # Required classrooms
    classrooms_2024 = math.ceil(
        pupils_2024 / CLASSROOM_CAPACITY
    )

    classrooms_2029 = math.ceil(
        pupils_2029 / CLASSROOM_CAPACITY
    )

    additional = (
        classrooms_2029
        - classrooms_2024
    )

    print(f"\n{district.name}")
    print(
        f"2024 population: "
        f"{population_2024:,.0f}"
    )

    print(
        f"2029 forecast: "
        f"{population_2029:,.0f}"
    )

    print(
        f"2029 primary-school-age population: "
        f"{pupils_2029:,.0f}"
    )

    print(
        f"Classrooms required in 2024: "
        f"{classrooms_2024}"
    )

    print(
        f"Classrooms required in 2029: "
        f"{classrooms_2029}"
    )

    print(
        f"Additional classrooms needed: "
        f"{additional}"
    )


# ============================================================
# EXTENSION: BOOTSTRAP 95% PREDICTION INTERVALS
# ============================================================

def bootstrap_prediction_interval(
    years,
    populations,
    horizon=5,
    n_bootstrap=1000
):

    # Fit linear model
    coefficients = np.polyfit(
        years,
        populations,
        1
    )

    fitted = np.polyval(
        coefficients,
        years
    )

    residuals = populations - fitted

    future_years = np.arange(
        years[-1] + 1,
        years[-1] + horizon + 1
    )

    rng = np.random.default_rng(42)

    predictions = []

    for _ in range(n_bootstrap):

        # Resample residuals
        sampled_residuals = rng.choice(
            residuals,
            size=len(residuals),
            replace=True
        )

        bootstrap_population = (
            fitted + sampled_residuals
        )

        # Refit
        bootstrap_coefficients = np.polyfit(
            years,
            bootstrap_population,
            1
        )

        # Predict
        future_prediction = np.polyval(
            bootstrap_coefficients,
            future_years
        )

        # Add prediction error
        future_errors = rng.choice(
            residuals,
            size=horizon,
            replace=True
        )

        future_prediction += future_errors

        predictions.append(
            future_prediction
        )

    predictions = np.array(predictions)

    lower = np.percentile(
        predictions,
        2.5,
        axis=0
    )

    upper = np.percentile(
        predictions,
        97.5,
        axis=0
    )

    return lower, upper


# ============================================================
# CRITICAL ASSESSMENT OF FIBONACCI MODEL
# ============================================================

print("\n" + "=" * 70)
print("FIBONACCI MODEL ASSESSMENT")
print("=" * 70)

print("""
The Fibonacci-ratio model is not a strong demographic forecasting
model.

It multiplies the previous population by successive Fibonacci ratios.
These ratios eventually approach the golden ratio:

1.618033...

Therefore, the model eventually implies population growth of roughly
61.8 percent per period.

Such growth is unrealistic for normal district population forecasting.
Population growth is affected by fertility, mortality, migration,
economic conditions, urbanisation and other demographic factors.

The Fibonacci model could only be considered defensible if there were
a strong theoretical or empirical reason for population changes to
follow Fibonacci ratios. In normal demographic forecasting, such a
relationship does not have a sound demographic justification.

Therefore, the Fibonacci model is best treated as an experimental
benchmark for this assignment rather than a realistic forecasting
method.
""")
