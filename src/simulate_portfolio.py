from pypm import metrics, signals, data_io, simulation

import pandas as pd
from typing import List

def simulate_portfolio():
    bollinger_n = 15
    sharpe_n = 30

    """ Load in data """
    symbols: List[str] = data_io.get_all_symbols()
    prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

    """ Use the Bollinger Band outer band crossover as a signal """
    _bollinger = signals.create_bollinger_band_signal
    signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)

    """ Use the sharpe_ratio as preference during collision """
    _sharpe = metrics.calculate_rolling_sharpe_ratio
    preference = prices.apply(_sharpe, args=(sharpe_n, ), axis=0)

    """ Run the simulator """
    simulator = simulation.SimpleSimulator(
        initial_cash=100000,
        max_active_positions=5,
        percent_slippage=0.0005,
        trade_fee=1,
    )
    simulator.simulate(prices, signal, preference)

    # Print results
    simulator.portfolio_history.print_position_summaries()
    simulator.print_initial_parameters()
    simulator.portfolio_history.print_summary()
    simulator.portfolio_history.plot_benchmark_comparison()

if __name__ == '__main__':
    simulate_portfolio()
