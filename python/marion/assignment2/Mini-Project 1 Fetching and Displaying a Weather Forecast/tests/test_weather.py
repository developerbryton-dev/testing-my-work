import json
import os
from src.weather_client import decode_weather_code, parse_forecast_to_dataframe

def load_sample(filename: str) -> dict:
    path = os.path.join(os.path.dirname(__file__), "sample_data", filename)
    with open(path, "r") as f:
        return json.load(f)

def test_decode_weather_code():
    assert decode_weather_code(0) == "Clear sky"
    assert decode_weather_code(61) == "Slight rain"
    assert "Unknown code" in decode_weather_code(999)

def test_parse_forecast_to_dataframe():
    forecast_data = load_sample("forecast_sample.json")
    df = parse_forecast_to_dataframe(forecast_data)
    
    assert len(df) == 2
    assert df.loc[0, "Condition"] == "Clear sky"
    assert df.loc[1, "Condition"] == "Slight rain"