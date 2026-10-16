# Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner

A reproducible Python/Jupyter project for analysing passenger demand on three Kampala matatu routes.

## Repository structure

- `notebooks/project5_taxi_planner.ipynb` — analysis and findings.
- `src/routes.py` — `Route` class, revenue and descriptive statistics.
- `src/forecasting.py` — forecasting base class, three required models, seasonal-naïve extension, walk-forward evaluation and alpha tuning.
- `src/equilibrium.py` — supply-demand equilibrium solved with `scipy.linalg.solve`.
- `tests/` — pytest unit tests.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Open the notebook from the repository root:

```bash
jupyter notebook notebooks/project5_taxi_planner.ipynb
```

Run all cells from top to bottom. The notebook inserts the repository root into `sys.path` so imports work when opened from the `notebooks/` folder.

Run tests from the repository root:

```bash
python -m pytest -q
```

## Assumptions

1. Passenger counts are total passenger boardings per route per day, and fare is charged per passenger per one-way trip.
2. Gross revenue is passenger count multiplied by fare. Fuel, wages, maintenance, empty return trips and other costs are not supplied, so profit is not estimated.
3. Descriptive variance and standard deviation use population definitions for the ten-day observed period.
4. Days 4–10 are evaluated using rolling-origin one-step forecasts. The forecast for each day uses only observations available before that day.
5. Exponential-smoothing alpha is selected separately for each route using a grid from 0.1 to 1.0 in increments of 0.1, scored over the same rolling origins.
6. Day-11 model selection uses the lowest route-specific walk-forward MAE among the moving average, tuned exponential smoothing and linear trend.
7. Fleet capacity assumes each vehicle completes eight one-way trips daily and can carry 14 passengers on each trip: 112 passenger places per vehicle-day. The 15% buffer is applied to forecast passenger demand and the number of vehicles is rounded up to a whole vehicle.
8. The 60-day extension is synthetic, generated with a fixed random seed. It has a weekly pattern with Friday demand higher and Sunday demand lower; it is not observed Kampala transport data.

## Main findings

The equilibrium model gives a fare of UGX 2,200 and quantity of 76 passengers per trip-hour. The current Ntinda fare of UGX 2,000 is UGX 200 below this modelled equilibrium. Route revenue and forecasting errors are calculated in the notebook from the supplied ten-day observations. Fleet recommendations depend on predicted demand, an assumed 112 passenger places per vehicle-day and a 15% buffer; they are planning estimates, not guarantees.

The extension compares seasonal-naïve forecasts with a three-day moving average on simulated data. The fixed seed makes the comparison reproducible. Because the data-generating process includes a weekly cycle, seasonal naïve is expected to exploit that structure, while a short moving average can blur the Friday and Sunday effects. The notebook prints the actual MAE comparison so this expectation can be checked.

## AI-use declaration

An AI coding assistant was used to help draft the project structure, Python classes, tests, notebook workflow and explanatory text. The student should run the notebook and tests, inspect each calculation, verify the results, and revise the analysis in their own words before submission. The student remains responsible for understanding and being able to explain all submitted code during the viva.
