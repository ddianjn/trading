import pandas as pd
from typing import List, Dict
from trading.data import data_fetching
from trading.backtest.position import Position
from trading.backtest.trade import Trade

def backtest(stocks: List[str]|str,
             start:str = "2018-01-01",
             end:str = None,
             interval:str = "1d",
             strategies: List = [],
             initial_capital: float = 100000,
             print_trades: bool = False,
             print_summary: bool = False) -> pd.DataFrame, List[Trade]:
  if isinstance(stocks, str):
    stocks = [stocks]
  res = []
  for stock in stocks:
    data = data_fetching.download_data(stock, start, end, interval = interval)
    data.reset_index(inplace=True)
    cash = initial_capital
    positions = []
    closed_positions = []
    transactions = []
    for i in range(2, len(data) - 2):
      for strategy in strategies:
        trade, position = strategy(stock, data, i, cash, positions)
        if trade is not None:
          transactions.append(trade)
          if print_trades:
            trade.print()
          assert position is not None
          if position.is_closed():
            assert position in positions
            positions.remove(position)
            closed_positions.append(position)
            cash += position.close_price * position.shares
          else:
            positions.append(position)
            cash -= position.open_price * position.shares
    end_price = data['Close'][len(data) - 1]
    end_date = data['Date'][len(data) - 1]
    for position in positions:
      trade = Trade(stock, end_date, end_price, position.shares, trade_type = 'Close', unit_cost = position.open_price)
      transactions.append(trade)
      position.close(end_date, end_price)
      closed_positions.append(position)
      positions.remove(position)
      cash += position.shares * end_price
      if print_trades:
        trade.print()
    res.append(summarize(stock, data, initial_capital, cash, transactions, closed_positions, print_summary))
  return pd.DataFrame(res), transactions

def summarize(stock: str,
              data: pd.DataFrame,
              initial_capital: float,
              final_capital: float,
              transactions: List[Trade],
              closed_positions: List[Position],
              print_summary: bool) -> Dict[str, float]:
  if print_summary:
    print(f"==={stock} Final Result===")
  total_return = final_capital - initial_capital
  percent_return = total_return / initial_capital
  # Annualize the return (assuming equal holding periods within a year)
  start_year = data['Date'][0].year
  end_year = data['Date'][len(data)-1].year
  holding_period = end_year - start_year + 1
  annualized_return = (1 + percent_return) ** (1 / holding_period) - 1
  
  if print_summary:
    print(f"% Return: {percent_return * 100: .2f}%")
    print(f"Annualized Return: {annualized_return * 100: .2f}%")

  res = {"Stock": stock,
         "% Return": percent_return,
         "Annualized Return":annualized_return}
  res.update(calculate_profit_trades(transactions, print_summary))
  res.update(calculate_profit_factor(closed_positions, print_summary))
  res.update(calculate_max_drawdown(initial_capital, transactions, print_summary))
  return res

def calculate_profit_trades(transactions: List[Trade],
                            print_summary: bool) -> Dict[str, float]:
  trade_with_profit = 0
  trade_with_loss = 0
  # Print profit trades
  for trade in transactions:
    if trade.trade_type == 'Close':
      if trade.price > trade.unit_cost:
        trade_with_profit += 1
      else:
        trade_with_loss += 1
  profit_rate = trade_with_profit / (trade_with_profit + trade_with_loss)
  if print_summary:
    print(f"Trade with Profit: {trade_with_profit: .2f}")
    print(f"Trade with loss: {trade_with_loss: .2f}")
    print(f"Profit Rate: {profit_rate * 100}%")
  return {"Trade with Profit": trade_with_profit,
          "Trade with loss": trade_with_loss,
          "Profit Rate": profit_rate}

def calculate_profit_factor(closed_positions: List[Position],
                            print_summary: bool) -> Dict[str, float]:
  gross_profit = 0
  gross_loss = 0
  for position in closed_positions:
    if position.profit > 0:
      gross_profit += position.profit
    else:
      gross_loss += position.profit
  profit_factor = gross_profit / -gross_loss
  if print_summary:
    print(f"Gross Profit: {gross_profit: .2f}")
    print(f"Gross Loss: {gross_loss: .2f}")
    print(f"Profit Factor: {profit_factor}")
  return {"Gross Profit": gross_profit,
          "Gross Loss": gross_loss,
          "Profit Factor": profit_factor}

def calculate_max_drawdown(initial_capital: float,
                           transactions: List[Trade],
                           print_summary: bool) -> Dict[str, float]:
  peak = initial_capital
  bottom = initial_capital
  drawdown = 0
  drawdown_percent = 0.0
  for trade in transactions:
    if trade.trade_type == 'Close':
      current = trade.price * trade.shares
      if current > peak:
        peak = current
      else:
        drawdown = max(drawdown, peak - current)
        drawdown_percent = max(drawdown_percent, (peak - current) / current)
  if print_summary:
    print(f"Max Drawdown: {drawdown: .2f}")
    print(f"Max % Drawdown: {drawdown_percent * 100: .2f}%")
  return {"Max Drawdown": drawdown, "Max % Drawdown": drawdown_percent}
