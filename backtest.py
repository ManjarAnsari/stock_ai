import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(df):
    df = df.copy()
    df['returns'] = df['Close'].pct_change().fillna(0)
    df['strategy_returns'] = df['returns'] * df['signal'].shift(1).fillna(0)

    cumulative_strategy = (1 + df['strategy_returns']).cumprod()
    cumulative_market = (1 + df['returns']).cumprod()

    trades = df['signal'].diff().abs().sum() / 2  # Buy/Sell counts
    avg_return = df['strategy_returns'].mean() * 100
    total_return = (cumulative_strategy.iloc[-1] - 1) * 100

    return {
        "returns": df['strategy_returns'],
        "cumulative_strategy": cumulative_strategy,
        "cumulative_market": cumulative_market,
        "trades": int(trades),
        "avg_return": avg_return,
        "total_return": total_return
    }

def plot_backtest_chart(df):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Close'], label='Close Price', color='blue', alpha=0.6)
    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]

    ax.plot(buy_signals.index, buy_signals['Close'], '^', markersize=10, color='green', label='Buy Signal')
    ax.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=10, color='red', label='Sell Signal')
    ax.set_title('Signal Overlay on Price')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

def plot_cumulative_returns(strategy_returns):
    cumulative_returns = (1 + strategy_returns).cumprod()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cumulative_returns, label='Cumulative Returns', color='purple')
    ax.set_title('Cumulative Strategy Returns')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    return fig
