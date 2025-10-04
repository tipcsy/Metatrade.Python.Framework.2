"""
Performance Calculator - Calculates comprehensive backtest metrics

This module computes a full suite of performance metrics including:
- Basic metrics: Win rate, profit factor, net profit
- Risk metrics: Maximum drawdown, Sharpe ratio, Sortino ratio
- Statistical metrics: Expectancy, recovery factor
- Trade analysis: MAE/MFE, trade duration, consecutive wins/losses
"""

import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PerformanceCalculator:
    """
    Calculates comprehensive performance metrics from backtest results.

    All formulas are based on industry-standard calculations used in
    professional trading platforms like MetaTrader, TradingView, etc.
    """

    def __init__(
        self,
        initial_balance: float,
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]]
    ):
        """
        Initialize performance calculator.

        Args:
            initial_balance: Starting account balance
            trades: List of trade records
            equity_curve: List of equity curve points
        """
        self.initial_balance = initial_balance
        self.trades = trades
        self.equity_curve = equity_curve

        # Convert to pandas for easier calculations
        self.trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_df = pd.DataFrame(equity_curve) if equity_curve else pd.DataFrame()

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Returns:
            Dictionary with all metrics
        """
        if len(self.trades) == 0:
            return self._get_empty_metrics()

        metrics = {}

        # Basic metrics
        metrics.update(self._calculate_basic_metrics())

        # Profit metrics
        metrics.update(self._calculate_profit_metrics())

        # Return metrics
        metrics.update(self._calculate_return_metrics())

        # Average metrics
        metrics.update(self._calculate_average_metrics())

        # Consecutive metrics
        metrics.update(self._calculate_consecutive_metrics())

        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics())

        # Risk-adjusted metrics
        metrics.update(self._calculate_risk_adjusted_metrics())

        # Statistical metrics
        metrics.update(self._calculate_statistical_metrics())

        # MAE/MFE metrics
        metrics.update(self._calculate_mae_mfe_metrics())

        # Duration metrics
        metrics.update(self._calculate_duration_metrics())

        # Account metrics
        metrics.update(self._calculate_account_metrics())

        # Commission metrics
        metrics.update(self._calculate_commission_metrics())

        return metrics

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "average_trade": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "max_drawdown": 0.0,
            "max_drawdown_duration_days": 0.0,
            "average_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sterling_ratio": 0.0,
            "expectancy": 0.0,
            "recovery_factor": 0.0,
            "average_mae": 0.0,
            "average_mfe": 0.0,
            "average_trade_duration_hours": 0.0,
            "initial_balance": self.initial_balance,
            "final_balance": self.initial_balance,
            "peak_balance": self.initial_balance,
            "total_commission": 0.0,
            "total_swap": 0.0
        }

    def _calculate_basic_metrics(self) -> Dict[str, Any]:
        """Calculate basic trade metrics"""
        df = self.trades_df

        total_trades = len(df)
        winning_trades = len(df[df['profit'] > 0])
        losing_trades = len(df[df['profit'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2)
        }

    def _calculate_profit_metrics(self) -> Dict[str, Any]:
        """Calculate profit-related metrics"""
        df = self.trades_df

        gross_profit = df[df['profit'] > 0]['profit'].sum()
        gross_loss = abs(df[df['profit'] < 0]['profit'].sum())
        net_profit = df['profit'].sum()

        # Profit factor = Gross profit / Gross loss
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

        return {
            "net_profit": round(net_profit, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 2)
        }

    def _calculate_return_metrics(self) -> Dict[str, Any]:
        """Calculate return metrics"""
        df = self.trades_df

        net_profit = df['profit'].sum()
        total_return = (net_profit / self.initial_balance) * 100

        # Calculate annualized return
        if len(self.equity_curve) > 0:
            first_time = self.equity_curve[0]['timestamp']
            last_time = self.equity_curve[-1]['timestamp']
            days = (last_time - first_time).total_seconds() / 86400

            if days > 0:
                years = days / 365.25
                annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0

        return {
            "total_return": round(total_return, 2),
            "annualized_return": round(annualized_return, 2)
        }

    def _calculate_average_metrics(self) -> Dict[str, Any]:
        """Calculate average trade metrics"""
        df = self.trades_df

        wins = df[df['profit'] > 0]['profit']
        losses = df[df['profit'] < 0]['profit']

        average_win = wins.mean() if len(wins) > 0 else 0.0
        average_loss = losses.mean() if len(losses) > 0 else 0.0
        average_trade = df['profit'].mean()
        largest_win = wins.max() if len(wins) > 0 else 0.0
        largest_loss = losses.min() if len(losses) > 0 else 0.0

        return {
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "average_trade": round(average_trade, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2)
        }

    def _calculate_consecutive_metrics(self) -> Dict[str, Any]:
        """Calculate consecutive wins/losses"""
        df = self.trades_df

        # Create win/loss sequence
        is_win = (df['profit'] > 0).astype(int)
        is_loss = (df['profit'] < 0).astype(int)

        # Count consecutive sequences
        max_consecutive_wins = self._max_consecutive(is_win.tolist())
        max_consecutive_losses = self._max_consecutive(is_loss.tolist())

        return {
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses
        }

    def _max_consecutive(self, sequence: List[int]) -> int:
        """Count maximum consecutive 1s in sequence"""
        if not sequence:
            return 0

        max_count = 0
        current_count = 0

        for value in sequence:
            if value == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count

    def _calculate_drawdown_metrics(self) -> Dict[str, Any]:
        """Calculate drawdown metrics"""
        if len(self.equity_df) == 0:
            return {
                "max_drawdown": 0.0,
                "max_drawdown_duration_days": 0.0,
                "average_drawdown": 0.0
            }

        # Calculate running maximum
        equity = self.equity_df['balance'].values
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown
        drawdown = (running_max - equity) / running_max * 100
        max_drawdown = drawdown.max()

        # Calculate drawdown duration
        max_dd_duration_days = self._calculate_max_drawdown_duration()

        # Calculate average drawdown
        drawdowns = drawdown[drawdown > 0]
        average_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0.0

        return {
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_duration_days": round(max_dd_duration_days, 2),
            "average_drawdown": round(average_drawdown, 2)
        }

    def _calculate_max_drawdown_duration(self) -> float:
        """Calculate maximum drawdown duration in days"""
        if len(self.equity_df) == 0:
            return 0.0

        equity = self.equity_df['balance'].values
        timestamps = self.equity_df['timestamp'].values

        running_max = np.maximum.accumulate(equity)
        in_drawdown = equity < running_max

        max_duration = 0
        current_duration = 0
        start_time = None

        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                if start_time is None:
                    start_time = timestamps[i]
                current_duration = (timestamps[i] - start_time).total_seconds() / 86400
                max_duration = max(max_duration, current_duration)
            else:
                start_time = None
                current_duration = 0

        return max_duration

    def _calculate_risk_adjusted_metrics(self) -> Dict[str, Any]:
        """Calculate risk-adjusted performance metrics"""
        # Sharpe Ratio
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Sortino Ratio
        sortino_ratio = self._calculate_sortino_ratio()

        # Calmar Ratio
        calmar_ratio = self._calculate_calmar_ratio()

        # Sterling Ratio
        sterling_ratio = self._calculate_sterling_ratio()

        return {
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "calmar_ratio": round(calmar_ratio, 2),
            "sterling_ratio": round(sterling_ratio, 2)
        }

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe Ratio.

        Sharpe Ratio = (Average Return - Risk-Free Rate) / Standard Deviation of Returns

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        if len(self.trades_df) == 0:
            return 0.0

        # Calculate trade returns
        returns = self.trades_df['profit'] / self.initial_balance

        if len(returns) < 2:
            return 0.0

        avg_return = returns.mean()
        std_return = returns.std()

        if std_return == 0:
            return 0.0

        # Annualize
        n_trades_per_year = self._estimate_trades_per_year()
        annualized_avg_return = avg_return * n_trades_per_year
        annualized_std = std_return * np.sqrt(n_trades_per_year)

        sharpe = (annualized_avg_return - risk_free_rate) / annualized_std

        return sharpe

    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio.

        Similar to Sharpe but only considers downside volatility.

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(self.trades_df) == 0:
            return 0.0

        returns = self.trades_df['profit'] / self.initial_balance
        negative_returns = returns[returns < 0]

        if len(negative_returns) < 2:
            return 0.0

        avg_return = returns.mean()
        downside_std = negative_returns.std()

        if downside_std == 0:
            return 0.0

        n_trades_per_year = self._estimate_trades_per_year()
        annualized_avg_return = avg_return * n_trades_per_year
        annualized_downside_std = downside_std * np.sqrt(n_trades_per_year)

        sortino = (annualized_avg_return - risk_free_rate) / annualized_downside_std

        return sortino

    def _calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar Ratio.

        Calmar Ratio = Annualized Return / Maximum Drawdown

        Returns:
            Calmar ratio
        """
        if len(self.trades_df) == 0:
            return 0.0

        net_profit = self.trades_df['profit'].sum()
        total_return = (net_profit / self.initial_balance) * 100

        # Calculate annualized return
        if len(self.equity_curve) > 0:
            first_time = self.equity_curve[0]['timestamp']
            last_time = self.equity_curve[-1]['timestamp']
            days = (last_time - first_time).total_seconds() / 86400

            if days > 0:
                years = days / 365.25
                annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
            else:
                return 0.0
        else:
            return 0.0

        # Get max drawdown
        if len(self.equity_df) == 0:
            return 0.0

        equity = self.equity_df['balance'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        max_drawdown = drawdown.max()

        if max_drawdown == 0:
            return 0.0

        calmar = annualized_return / max_drawdown

        return calmar

    def _calculate_sterling_ratio(self) -> float:
        """
        Calculate Sterling Ratio.

        Sterling Ratio = Annualized Return / Average Drawdown

        Returns:
            Sterling ratio
        """
        if len(self.trades_df) == 0 or len(self.equity_df) == 0:
            return 0.0

        net_profit = self.trades_df['profit'].sum()
        total_return = (net_profit / self.initial_balance) * 100

        # Calculate annualized return
        if len(self.equity_curve) > 0:
            first_time = self.equity_curve[0]['timestamp']
            last_time = self.equity_curve[-1]['timestamp']
            days = (last_time - first_time).total_seconds() / 86400

            if days > 0:
                years = days / 365.25
                annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
            else:
                return 0.0
        else:
            return 0.0

        # Get average drawdown
        equity = self.equity_df['balance'].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max * 100
        drawdowns = drawdown[drawdown > 0]
        average_drawdown = drawdowns.mean() if len(drawdowns) > 0 else 0.0

        if average_drawdown == 0:
            return 0.0

        sterling = annualized_return / average_drawdown

        return sterling

    def _estimate_trades_per_year(self) -> float:
        """Estimate number of trades per year based on historical data"""
        if len(self.equity_curve) < 2:
            return 252.0  # Default to daily trading

        first_time = self.equity_curve[0]['timestamp']
        last_time = self.equity_curve[-1]['timestamp']
        days = (last_time - first_time).total_seconds() / 86400

        if days == 0:
            return 252.0

        trades_per_day = len(self.trades_df) / days
        return trades_per_day * 365.25

    def _calculate_statistical_metrics(self) -> Dict[str, Any]:
        """Calculate statistical metrics"""
        if len(self.trades_df) == 0:
            return {
                "expectancy": 0.0,
                "recovery_factor": 0.0
            }

        # Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
        df = self.trades_df
        wins = df[df['profit'] > 0]['profit']
        losses = df[df['profit'] < 0]['profit']

        win_rate = len(wins) / len(df) if len(df) > 0 else 0.0
        loss_rate = 1 - win_rate

        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Recovery Factor = Net Profit / Maximum Drawdown (in dollars)
        net_profit = df['profit'].sum()

        if len(self.equity_df) > 0:
            equity = self.equity_df['balance'].values
            running_max = np.maximum.accumulate(equity)
            drawdown_dollars = (running_max - equity).max()

            recovery_factor = abs(net_profit / drawdown_dollars) if drawdown_dollars > 0 else 0.0
        else:
            recovery_factor = 0.0

        return {
            "expectancy": round(expectancy, 2),
            "recovery_factor": round(recovery_factor, 2)
        }

    def _calculate_mae_mfe_metrics(self) -> Dict[str, Any]:
        """Calculate MAE/MFE metrics"""
        if len(self.trades_df) == 0:
            return {
                "average_mae": 0.0,
                "average_mfe": 0.0
            }

        df = self.trades_df

        # Filter out None values
        mae_values = df[df['mae'].notna()]['mae']
        mfe_values = df[df['mfe'].notna()]['mfe']

        average_mae = mae_values.mean() if len(mae_values) > 0 else 0.0
        average_mfe = mfe_values.mean() if len(mfe_values) > 0 else 0.0

        return {
            "average_mae": round(average_mae, 2),
            "average_mfe": round(average_mfe, 2)
        }

    def _calculate_duration_metrics(self) -> Dict[str, Any]:
        """Calculate trade duration metrics"""
        if len(self.trades_df) == 0:
            return {
                "average_trade_duration_hours": 0.0
            }

        df = self.trades_df

        # Filter out None values
        durations = df[df['duration_seconds'].notna()]['duration_seconds']

        average_duration_hours = (durations.mean() / 3600) if len(durations) > 0 else 0.0

        return {
            "average_trade_duration_hours": round(average_duration_hours, 2)
        }

    def _calculate_account_metrics(self) -> Dict[str, Any]:
        """Calculate account-related metrics"""
        final_balance = self.initial_balance

        if len(self.equity_df) > 0:
            final_balance = self.equity_df['balance'].iloc[-1]
            peak_balance = self.equity_df['balance'].max()
        else:
            peak_balance = self.initial_balance

        return {
            "initial_balance": round(self.initial_balance, 2),
            "final_balance": round(final_balance, 2),
            "peak_balance": round(peak_balance, 2)
        }

    def _calculate_commission_metrics(self) -> Dict[str, Any]:
        """Calculate commission and swap metrics"""
        if len(self.trades_df) == 0:
            return {
                "total_commission": 0.0,
                "total_swap": 0.0
            }

        df = self.trades_df

        total_commission = df['commission'].sum()
        total_swap = df['swap'].sum()

        return {
            "total_commission": round(total_commission, 2),
            "total_swap": round(total_swap, 2)
        }
