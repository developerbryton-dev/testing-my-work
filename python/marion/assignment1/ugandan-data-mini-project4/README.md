Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser
Overview

This project analyses monthly rainfall patterns for Kampala, Gulu and Mbarara in Uganda. It compares rainfall regimes, evaluates simplified crop suitability, measures similarity between regions, detects rainfall peaks, and provides a farmer advisory.

The rainfall data supplied in the assignment are illustrative synthetic data. Therefore, the results demonstrate the analytical methods and should not be interpreted as official agricultural advice.

Project Structure
ugandan-data-mini-projects/
├── Notebooks/
│   └── project4_rainfall_crop_suitability.ipynb
├── src/
│   ├── __init__.py
│   └── rainfall_model.py
├── tests/
│   └── test_rainfall_model.py
└── README.md

Requirements

The project uses Python with the following main libraries:

numpy

pandas

scipy

matplotlib

pytest

Install the dependencies with:

python -m pip install numpy pandas scipy matplotlib pytest

Running the Notebook

Open Jupyter Notebook from the project directory:

jupyter notebook


Then open:

Notebooks/project4_rainfall_crop_suitability.ipynb


Run the notebook from beginning to end using Restart & Run All.

Running the Tests

From the project root, run:

python -m pytest


The current test suite contains five tests covering rainfall calculations, invalid/empty input, crop classification and cosine similarity.

Main Methods

The project implements:

A Region class for rainfall statistics.

A CropRule class for monthly crop suitability.

Correct vector cosine similarity using NumPy.

Verification against SciPy's cosine distance calculation.

Pearson correlation and Euclidean distance.

Automatic rainfall peak detection using scipy.signal.find_peaks.

Rainfall and crop-suitability visualisations.

A farmer advisory based on the illustrative rainfall data.

Findings Summary

The illustrative rainfall patterns differ considerably between the three regions. Kampala has a dominant rainfall peak around May and is classified as unimodal by the peak-detection method. Gulu has important peaks around May and August, while Mbarara has peaks around April and October, producing a bimodal classification.

Cosine similarity and Pearson correlation measure the similarity in rainfall patterns, while Euclidean distance is more sensitive to the absolute magnitude of rainfall. Therefore, two regions can have similar seasonal shapes even when one receives substantially more rainfall.

The crop-suitability results demonstrate how monthly rainfall information can support basic planting decisions. However, the thresholds used are simplified and should not be treated as universal agronomic recommendations.

Limitations

The supplied dataset represents only one illustrative year. It therefore cannot measure year-to-year rainfall variability, long-term climate trends or drought frequency. The analysis also does not include temperature, soil type, evapotranspiration, soil moisture, pests or crop varieties.

For a stronger agricultural decision-support system, the analysis should be repeated using at least 10 years of observed or satellite-derived rainfall data, such as NASA POWER or suitable Ugandan meteorological data.

AI Use Declaration

AI assistance was used to help explain the assignment requirements, structure the project, develop and debug parts of the Python code, create unit-test examples, and explain errors encountered while running the notebook and tests. The final code was run and tested by the student, and the student is responsible for understanding and explaining the submitted work.