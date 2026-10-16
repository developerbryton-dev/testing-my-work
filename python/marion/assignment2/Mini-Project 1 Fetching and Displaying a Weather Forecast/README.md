# Mini-Project 1: Fetching and Displaying a Weather Forecast

## Overview
This project is a robust Python application that retrieves geographic coordinates from a location name using the Open-Meteo Geocoding API, fetches a 7-day weather forecast, processes the data using Pandas, and visualizes temperature ranges and precipitation using Matplotlib.

---

## Repository Structure
- `src/`: Contains reusable Python modules (`weather_client.py` for API requests, error handling, and data parsing).
- `notebooks/`: Contains the executable Jupyter Notebook (`project1_weather.ipynb`) covering Parts A through D.
- `tests/`: Contains offline unit tests (`test_weather.py`) and saved sample JSON files for verification without an internet connection.

---

## Setup and Installation

1. **Clone or Download the Repository** to your local machine.
2. **Install Dependencies:**
   Open your terminal/command prompt in the project root folder and run:
   ```bash
   pip install -r requirements.txt