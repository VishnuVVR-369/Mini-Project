import pandas as pd
from pypm.indicators import calculate_bollinger_bands


def create_bollinger_band_signal(series: pd.Series, n: int=20) -> pd.Series:
    """
    Create signals based on the upper and lower bands of the 
    Bollinger bands. Generate a buy signal when the price is below the lower 
    band, and a sell signal when the price is above the upper band.
    """
    bollinger_bands = calculate_bollinger_bands(series, n)
    sell = series > bollinger_bands['upper']
    buy = series < bollinger_bands['lower']
    return (1*buy - 1*sell)


""" Check if functions execute successfully """
# create_bollinger_band_signal