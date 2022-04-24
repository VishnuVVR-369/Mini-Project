import os
import pandas as pd
from pandas import DataFrame
from typing import Dict, List

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    '..',
    '..',
    'data',
)
EOD_DATA_DIR = os.path.join(DATA_DIR, 'eod')


""" Loading 50 stocks data """
def load_eod_data(ticker: str, data_dir: str=EOD_DATA_DIR) -> DataFrame:
    f_path = os.path.join(data_dir, f'{ticker}.csv')
    assert os.path.isfile(f_path), f'No data available for {ticker}'
    return pd.read_csv(f_path, parse_dates=['date'], index_col='date')


def load_nifty_data() -> DataFrame:
    """ Loading NIFTY 50 data using load_eod_data function """
    return load_eod_data('NIFTY 50', DATA_DIR)


def _combine_columns(filepaths_by_symbol: Dict[str, str], attr: str='close') -> pd.DataFrame:
    data_frames = [
        pd.read_csv(
            filepath,
            index_col='date',
            usecols=['date', attr],
            parse_dates=['date'],
        ).rename(
            columns={
                'date': 'date',
                attr: symbol,
            }
        ) for symbol, filepath in filepaths_by_symbol.items()
    ]
    return pd.concat(data_frames, sort=True, axis=1)


def load_eod_matrix(tickers: List[str], attr: str='close') -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(EOD_DATA_DIR, f'{t}.csv') for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, attr)


def get_all_symbols() -> List[str]:
    """ Get symbols of stocks """
    return [v.strip('.csv') for v in os.listdir(EOD_DATA_DIR)]



def concatenate_metrics(df_by_metric: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates different dataframes that have the same columns into a
    hierarchical dataframe.
    The input df_by_metric should be of the form
    {
        'metric_1': pd.DataFrame()
        'metric_2: pd.DataFrame()
    }
    where each dataframe should have the same symbols.
    """
    to_concatenate = []
    tuples = []
    for key, df in df_by_metric.items():
        to_concatenate.append(df)
        tuples += [(s, key) for s in df.columns.values]

    df = pd.concat(to_concatenate, sort=True, axis=1)
    df.columns = pd.MultiIndex.from_tuples(tuples, names=['symbol', 'metric'])

    return df


""" Check if functions execute successfully """
# load_eod_data
# load_nifty_data
# load_eod_matrix
# get_all_symbols
# concatenate_metrics
