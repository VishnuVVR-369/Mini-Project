import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, NewType, Any, Set
from collections import OrderedDict, defaultdict

from pypm import metrics, data_io

Symbol = NewType('Symbol', str)
rupees = NewType('rupees', float)

DATE_FORMAT_STR = '%a %b %d, %Y'

File = open("../Trade Summary.txt", "w", encoding="UTF-8")

def _pdate(date: pd.Timestamp):
    """ Pretty-print a datetime with just the date """
    return date.strftime(DATE_FORMAT_STR)


class Position(object):

    def __init__(self, symbol: Symbol, entry_date: pd.Timestamp, entry_price: rupees, shares: int):

        self.entry_date = entry_date

        assert entry_price > 0, 'Cannot buy asset with zero or negative price.'
        self.entry_price = entry_price

        assert shares > 0, 'Cannot buy zero or negative shares.'
        self.shares = shares

        self.symbol = symbol

        self.exit_date: pd.Timestamp = None
        self.exit_price: rupees = None

        self.last_date: pd.Timestamp = None
        self.last_price: rupees = None

        self._dict_series: Dict[pd.Timestamp, rupees] = OrderedDict()
        self.record_price_update(entry_date, entry_price)

        self._price_series: pd.Series = None
        self._needs_update_pd_series: bool = True


    def exit(self, exit_date, exit_price):
        assert self.entry_date != exit_date, 'Churned a position same-day.'
        assert not self.exit_date, 'Position already closed.'
        self.record_price_update(exit_date, exit_price)
        self.exit_date = exit_date
        self.exit_price = exit_price

    def record_price_update(self, date, price):
        self.last_date = date
        self.last_price = price
        self._dict_series[date] = price

        self._needs_update_pd_series = True

    @property
    def price_series(self) -> pd.Series:
        if self._needs_update_pd_series or self._price_series is None:
            self._price_series = pd.Series(self._dict_series)
            self._needs_update_pd_series = False
        return self._price_series

    @property
    def last_value(self) -> rupees:
        return self.last_price * self.shares

    @property
    def is_active(self) -> bool:
        return self.exit_date is None

    @property
    def is_closed(self) -> bool:
        return not self.is_active

    @property
    def value_series(self) -> pd.Series:
        assert self.is_closed, 'Position must be closed to access this property'
        return self.shares * self.price_series[:-1]

    @property
    def percent_return(self) -> float:
        return (self.exit_price / self.entry_price) - 1

    @property
    def entry_value(self) -> rupees:
        return self.shares * self.entry_price

    @property
    def exit_value(self) -> rupees:
        return self.shares * self.exit_price

    @property
    def change_in_value(self) -> rupees:
        return self.exit_value - self.entry_value

    @property
    def trade_length(self):
        return len(self._dict_series) - 1

    def print_position_summary(self):
        _entry_date = _pdate(self.entry_date)
        _exit_date = _pdate(self.exit_date)
        _days = self.trade_length

        _entry_price = round(self.entry_price, 2)
        _exit_price = round(self.exit_price, 2)

        _entry_value = round(self.entry_value, 2)
        _exit_value = round(self.exit_value, 2)

        _return = round(100 * self.percent_return, 1)
        _diff = round(self.change_in_value, 2)

        File.write(f'{self.symbol:<5}     Trade summary\n')
        File.write(f'Date:     {_entry_date} -> {_exit_date} [{_days} days]\n')
        File.write(f'Price:    ₹{_entry_price} -> ₹{_exit_price} [{_return}%]\n')
        File.write(f'Value:    ₹{_entry_value} -> ₹{_exit_value} [₹{_diff}]\n')
        File.write("\n")
        # print(f'{self.symbol:<5}     Trade summary')
        # print(f'Date:     {_entry_date} -> {_exit_date} [{_days} days]')
        # print(f'Price:    ₹{_entry_price} -> ₹{_exit_price} [{_return}%]')
        # print(f'Value:    ₹{_entry_value} -> ₹{_exit_value} [₹{_diff}]')
        # print()

    def __hash__(self):
        return hash((self.entry_date, self.symbol))

    
class PortfolioHistory(object):

    def __init__(self):
        self.position_history: List[Position] = []
        self._logged_positions: Set[Position] = set()

        self.last_date: pd.Timestamp = pd.Timestamp.min

        self._cash_history: Dict[pd.Timestamp, rupees] = dict()
        self._simulation_finished = False
        self._nifty: pd.DataFrame = pd.DataFrame()
        self._nifty_log_returns: pd.Series = pd.Series(dtype='object')

    def add_to_history(self, position: Position):
        _log = self._logged_positions
        assert not position in _log, 'Recorded the same position twice.'
        assert position.is_closed, 'Position is not closed.'
        self._logged_positions.add(position)
        self.position_history.append(position)
        self.last_date = max(self.last_date, position.last_date)

    def record_cash(self, date, cash):
        self._cash_history[date] = cash
        self.last_date = max(self.last_date, date)

    @staticmethod
    def _as_oseries(d: Dict[pd.Timestamp, Any]) -> pd.Series:
        return pd.Series(d).sort_index()

    def _compute_cash_series(self):
        self._cash_series = self._as_oseries(self._cash_history)

    @property
    def cash_series(self) -> pd.Series:
        return self._cash_series

    def _compute_portfolio_value_series(self):
        value_by_date = defaultdict(float)
        last_date = self.last_date

        for position in self.position_history:
            for date, value in position.value_series.items():
                value_by_date[date] += value

        for date in self.cash_series.index:
            value_by_date[date] += 0

        self._portfolio_value_series = self._as_oseries(value_by_date)

    @property
    def portfolio_value_series(self):
        return self._portfolio_value_series

    def _compute_equity_series(self):
        c_series = self.cash_series
        p_series = self.portfolio_value_series
        assert all(c_series.index == p_series.index), \
            'portfolio_series has dates not in cash_series'
        self._equity_series = c_series + p_series

    @property
    def equity_series(self):
        return self._equity_series

    def _compute_log_return_series(self):
        self._log_return_series = metrics.calculate_log_return_series(self.equity_series)

    @property
    def log_return_series(self):
        return self._log_return_series

    def _assert_finished(self):
        assert self._simulation_finished, \
            'Simulation must be finished by running self.finish() in order ' + \
            'to access this method or property.'

    def finish(self):
        self._simulation_finished = True
        self._compute_cash_series()
        self._compute_portfolio_value_series()
        self._compute_equity_series()
        self._compute_log_return_series()
        self._assert_finished()

    def compute_portfolio_size_series(self) -> pd.Series:
        size_by_date = defaultdict(int)
        for position in self.position_history:
            for date in position.value_series.index:
                size_by_date[date] += 1
        return self._as_oseries(size_by_date)

    @property
    def nifty(self):
        if self._nifty.empty:
            first_date = self.cash_series.index[0]
            _nifty = data_io.load_nifty_data()
            self._nifty = _nifty[_nifty.index > first_date]
        return self._nifty

    @property
    def nifty_log_returns(self):
        if self._nifty_log_returns.empty:
            close = self.nifty['close']
            self._nifty_log_returns = metrics.calculate_log_return_series(close)
        return self._nifty_log_returns

    @property
    def percent_return(self):
        return metrics.calculate_percent_return(self.equity_series)

    @property
    def nifty_percent_return(self):
        return metrics.calculate_percent_return(self.nifty['close'])

    @property
    def cagr(self):
        return metrics.calculate_cagr(self.equity_series)

    @property
    def volatility(self):
        return metrics.calculate_annualized_volatility(self.log_return_series)

    @property
    def sharpe_ratio(self):
        return metrics.calculate_sharpe_ratio(self.equity_series)

    @property
    def nifty_cagr(self):
        return metrics.calculate_cagr(self.nifty['close'])

    @property
    def excess_cagr(self):
        return self.cagr - self.nifty_cagr

    @property
    def jensens_alpha(self):
        return metrics.calculate_jensens_alpha(
            self.log_return_series,
            self.nifty_log_returns,
        )

    @property
    def rupee_max_drawdown(self):
        return metrics.calculate_max_drawdown(self.equity_series, 'rupee')

    @property
    def percent_max_drawdown(self):
        return metrics.calculate_max_drawdown(self.equity_series, 'percent')

    @property
    def log_max_drawdown_ratio(self):
        return metrics.calculate_log_max_drawdown_ratio(self.equity_series)

    @property
    def number_of_trades(self):
        return len(self.position_history)

    @property
    def average_active_trades(self):
        return self.compute_portfolio_size_series().mean()

    @property
    def final_cash(self):
        self._assert_finished()
        return self.cash_series[-1]

    @property
    def final_equity(self):
        self._assert_finished()
        return self.equity_series[-1]

    _PERFORMANCE_METRICS_PROPS = [
        'percent_return',
        'nifty_percent_return',
        'cagr',
        'volatility',
        'sharpe_ratio',
        'nifty_cagr',
        'excess_cagr',
        'jensens_alpha',
        'rupee_max_drawdown',
        'percent_max_drawdown',
        'log_max_drawdown_ratio',
        'number_of_trades',
        'average_active_trades',
        'final_cash',
        'final_equity',
    ]

    PerformancePayload = NewType('PerformancePayload', Dict[str, float])

    def get_performance_metric_data(self) -> PerformancePayload:
        props = self._PERFORMANCE_METRICS_PROPS
        return {prop: getattr(self, prop) for prop in props}

    def print_position_summaries(self):
        for position in self.position_history:
            position.print_position_summary()

    def print_summary(self):
        self._assert_finished()
        s = f'Equity: ₹{self.final_equity:.2f}\n' \
            f'Percent Return: {100 * self.percent_return:.2f}%\n' \
            f'NIFTY 50 Return: {100 * self.nifty_percent_return:.2f}%\n\n' \
            f'Number of trades: {self.number_of_trades}\n' \
            f'Average active trades: {self.average_active_trades:.2f}\n\n' \
            f'CAGR: {100 * self.cagr:.2f}%\n' \
            f'NIFTY 50 CAGR: {100 * self.nifty_cagr:.2f}%\n' \
            f'Excess CAGR: {100 * self.excess_cagr:.2f}%\n\n' \
            # f'Annualized Volatility: {100 * self.volatility:.2f}%\n' \
            # f'Sharpe Ratio: {self.sharpe_ratio:.2f}\n' \
            # f'Jensen\'s Alpha: {self.jensens_alpha:.6f}\n\n' \
            # f'Rupee Max Drawdown: ₹{self.rupee_max_drawdown:.2f}\n' \
            # f'Percent Max Drawdown: {100 * self.percent_max_drawdown:.2f}%\n' \
            # f'Log Max Drawdown Ratio: {self.log_max_drawdown_ratio:.2f}\n'

        print(s)

    def plot(self, show=True) -> plt.Figure:
        """ Plots equity, cash and portfolio value curves """
        self._assert_finished()

        figure, axes = plt.subplots(nrows=3, ncols=1)
        figure.tight_layout(pad=3.0)
        axes[0].plot(self.equity_series)
        axes[0].set_title('Equity')
        axes[0].grid()

        axes[1].plot(self.cash_series)
        axes[1].set_title('Cash')
        axes[1].grid()

        axes[2].plot(self.portfolio_value_series)
        axes[2].set_title('Portfolio Value')
        axes[2].grid()

        if show:
            plt.show()

        return figure

    def plot_benchmark_comparison(self, show=True) -> plt.Figure:
        self._assert_finished()

        equity_curve = self.equity_series
        ax = equity_curve.plot()

        nifty_closes = self.nifty['close']
        initial_cash = self.cash_series[0]
        initial_nifty = nifty_closes[0]

        scaled_nifty = nifty_closes * (initial_cash / initial_nifty)
        scaled_nifty.plot()

        baseline = pd.Series(initial_cash, index=equity_curve.index)
        ax = baseline.plot(color='black')
        ax.grid()

        ax.legend(['Equity curve', 'NIFTY 50 portfolio'])

        if show:
            plt.show()

