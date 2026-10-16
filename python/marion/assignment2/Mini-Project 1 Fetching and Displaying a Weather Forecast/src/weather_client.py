import time
import requests
import pandas as pd
from typing import Dict, Any, List, Optional

class APIRequestError(Exception):
    """Custom exception raised when an API request fails permanently."""
    pass

def robust_get(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, 
               timeout: float = 5.0, max_retries: int = 3, backoff_factor: float = 1.5) -> requests.Response:
    """
    Performs an HTTP GET request with timeouts, raise_for_status, and exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                raise APIRequestError(f"Network error after {max_retries} attempts: {e}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            if 500 <= status < 600 and attempt < max_retries:
                pass
            else:
                raise APIRequestError(f"HTTP error {status}: {e}")
        
        sleep_time = backoff_factor ** attempt
        time.sleep(sleep_time)
        
    raise APIRequestError("Failed to fetch data due to unknown network issues.")

def geocode(name: str, contact_email: str) -> List[Dict[str, Any]]:
    """
    Geocodes a location name using Open-Meteo Geocoding API.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    headers = {"User-Agent": f"StudentProject ({contact_email})"}
    params = {"name": name, "count": 10}
    
    response = robust_get(url, params=params, headers=headers)
    data = response.json()
    
    return data.get("results", [])

def get_forecast(lat: float, lon: float, days: int, contact_email: str) -> Dict[str, Any]:
    """
    Retrieves daily weather forecast from Open-Meteo API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    headers = {"User-Agent": f"StudentProject ({contact_email})"}
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "timezone": "auto",
        "forecast_days": days
    }
    
    response = robust_get(url, params=params, headers=headers)
    return response.json()

def decode_weather_code(code: int) -> str:
    """
    Decodes Open-Meteo WMO weather interpretation codes into human-readable descriptions.
    """
    weather_mapping = {
        0: "Clear sky",
        1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
        95: "Thunderstorm: Slight or moderate",
        99: "Thunderstorm with heavy hail"
    }
    return weather_mapping.get(code, f"Unknown code ({code})")

def parse_forecast_to_dataframe(forecast_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Parses Open-Meteo JSON response into a clean Pandas DataFrame.
    """
    daily = forecast_json.get("daily", {})
    df = pd.DataFrame({
        "Date": daily.get("time", []),
        "Weather Code": daily.get("weathercode", []),
        "Max Temp (°C)": daily.get("temperature_2m_max", []),
        "Min Temp (°C)": daily.get("temperature_2m_min", []),
        "Precipitation (mm)": daily.get("precipitation_sum", []),
        "Max Wind Speed (km/h)": daily.get("windspeed_10m_max", [])
    })
    
    df["Condition"] = df["Weather Code"].apply(decode_weather_code)
    return df