import numpy as np
import pandas as pd
from pypm.data_io import load_nifty_data
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Callable

def calculate_return_series(series: pd.Series) -> pd.Series:
    """
    Calculates the return series of a given time series.
    The first value will always be NaN.
    """
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1


def calculate_log_return_series(series: pd.Series) -> pd.Series:
    """
    Same as calculate_return_series but with log returns
    """
    shifted_series = series.shift(1, axis=0)
    return pd.Series(np.log(series / shifted_series))


def calculate_percent_return(series: pd.Series) -> float:
    """
    Takes the first and last value in a series to determine the percent return
    """
    return series.iloc[-1] / series.iloc[0] - 1


def get_years_past(series: pd.Series) -> float:
    """
    Calculate the years past for use with functions that require annualization   
    """
    start_date = series.index[0]
    end_date = series.index[-1]
    return (end_date - start_date).days / 365.25


def calculate_cagr(series: pd.Series) -> float:
    """
    Calculating compounded annual growth rate
    """
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    value_factor = end_price / start_price
    year_past = get_years_past(series)
    return (value_factor ** (1 / year_past)) - 1


def calculate_annualized_volatility(return_series: pd.Series) -> float:
    """
    Calculating annualized volatility for a date-indexed return series. 
    """
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past
    return return_series.std() * np.sqrt(entries_per_year)


def calculate_sharpe_ratio(price_series: pd.Series, benchmark_rate: float=0) -> float:
    """
    Calculates the Sharpe ratio given a price series.
    """
    cagr = calculate_cagr(price_series)
    return_series = calculate_return_series(price_series)
    volatility = calculate_annualized_volatility(return_series)
    return (cagr - benchmark_rate) / volatility


def calculate_rolling_sharpe_ratio(price_series: pd.Series, n: float=20) -> pd.Series:
    """
    Compute an approximation of the sharpe ratio on a rolling basis. 
    This is used as a preference value when conflict occurs.
    """
    rolling_return_series = calculate_return_series(price_series).rolling(n)
    return rolling_return_series.mean() / rolling_return_series.std()


def calculate_jensens_alpha(return_series: pd.Series, benchmark_return_series: pd.Series) -> float:
    """
    Calculates Jensen's alpha. Prefers input series have the same index.
    """
    # Join series along date index and purge NAs
    df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
    df = df.dropna()

    # Get the appropriate data structure for scikit learn
    clean_returns: pd.Series = df[df.columns.values[0]]
    clean_benchmarks = pd.DataFrame(df[df.columns.values[1]])

    # Fit a linear regression and return the alpha
    regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
    return regression.intercept_


DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
    'rupee': lambda price, peak: peak - price,
    'percent': lambda price, peak: -((price / peak) - 1),
    'log': lambda price, peak: np.log(peak) - np.log(price),
}


def calculate_drawdown_series(series: pd.Series, method: str='log') -> pd.Series:
    """
    Returns the drawdown series
    """
    assert method in DRAWDOWN_EVALUATORS, \
        f'Method "{method}" must by one of {list(DRAWDOWN_EVALUATORS.keys())}'

    evaluator = DRAWDOWN_EVALUATORS[method]
    return evaluator(series, series.cummax())


def calculate_max_drawdown(series: pd.Series, method: str='log') -> float:
    """
    Simply returns the max drawdown as a float
    """
    return calculate_drawdown_series(series, method).max()


def calculate_log_max_drawdown_ratio(series: pd.Series) -> float:
    log_drawdown = calculate_max_drawdown(series, method='log')
    log_return = np.log(series.iloc[-1]) - np.log(series.iloc[0])
    return log_return - log_drawdown
