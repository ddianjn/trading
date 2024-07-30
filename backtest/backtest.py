import pandas as pd
from typing import List, Dict
from trading.data import data_fetching
from trading.backtest.position import Position
from trading.backtest.strategy import Strategy
from trading.backtest.trade import Trade
from trading import plotting

def backtest(stocks: List[str]|str,
             start:str = "2018-01-01",
             end:str = None,
             interval:str = "1d",
             strategies: List[Strategy] = [],
             initial_capital: float = 100000,
             print_trades: bool = False,
             print_balance: bool = False,
             print_summary: bool = False,
             plot_net_values: bool = False) -> (pd.DataFrame, List[Trade]):
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
    net_values = []
    for strategy in strategies:
      data = strategy.generate_indicator_columns(data)
    for i in range(len(data)):
      has_trade = False
      for strategy in strategies:
        trades = strategy.trade(stock, data, i, cash, positions)
        for trade in trades:
          has_trade = True
          transactions.append(trade)
          if print_trades:
            trade.print()
          position = trade.affected_position
          if trade.is_close():
            closed_positions.append(position)
            cash += position.close_price * position.shares
          else:
            cash -= position.open_price * position.shares
      net_values.append(calculate_net_value(stock, data, cash, positions))
      if has_trade and print_balance:
        print(f"{data['Date'][i]}: Cash = {cash}, Net value = {net_values[-1]['Close']}")
    end_price = data['Close'][len(data) - 1]
    end_date = data['Date'][len(data) - 1]
    for position in positions:
      trade = Trade(stock, end_date, end_price, position.shares, trade_type = 'Close', unit_cost = position.open_price)
      transactions.append(trade)
      position.close(end_date, end_price)
      closed_positions.append(position)
      positions.remove(position)
      cash += position.shares * end_price
      net_values.append(calculate_net_value(stock, data, cash, positions))
      if print_trades:
        trade.print()
    res.append(summarize(stock, data, initial_capital, cash, transactions, closed_positions, net_values, print_summary, plot_net_values))
  return pd.DataFrame(res), transactions

def calculate_net_value(stock: str,
                        data: pd.DataFrame,
                        cash: int,
                        positions: List[Position]):
  high_net_value = cash
  low_net_value = cash
  close_net_value = cash
  for position in positions:
    if position.stock == stock:
      high_net_value += data['High'] * position.shares
      low_net_value += data['Low'] * position.shares
      close_net_value += data['Close'] * position.shares
  return {"High": high_net_value, "Low": low_net_value, "Close": close_net_value}

def summarize(stock: str,
              data: pd.DataFrame,
              initial_capital: float,
              final_capital: float,
              transactions: List[Trade],
              closed_positions: List[Position],
              net_values: List[Dict[str, float]],
              print_summary: bool,
              plot_net_values: bool) -> Dict[str, float]:
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
         "% Annualized Return":annualized_return}
  res.update(calculate_profit_trades(transactions, print_summary))
  res.update(calculate_profit_factor(closed_positions, print_summary))
  res.update(calculate_max_drawdown(net_values, print_summary))
  if plot_net_values:
    plot_net_values_chart(stock, data, net_values)
  return res

def calculate_profit_trades(transactions: List[Trade],
                            print_summary: bool) -> Dict[str, float]:
  trade_with_profit = 0
  trade_with_loss = 0
  # Print profit trades
  for trade in transactions:
    if trade.is_close():
      if trade.price > trade.unit_cost:
        trade_with_profit += 1
      else:
        trade_with_loss += 1
  if trade_with_profit == 0:
    profit_rate = 0
  else:
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
  if gross_loss == 0:
    profit_factor = float('inf')
  else:
    profit_factor = gross_profit / -gross_loss
  if print_summary:
    print(f"Gross Profit: {gross_profit: .2f}")
    print(f"Gross Loss: {gross_loss: .2f}")
    print(f"Profit Factor: {profit_factor}")
  return {"Gross Profit": gross_profit,
          "Gross Loss": gross_loss,
          "Profit Factor": profit_factor}

def calculate_max_drawdown(net_values: List[Dict[str, float]],
                           print_summary: bool) -> Dict[str, float]:
  peak = net_values[0]['High']
  drawdown = 0
  drawdown_percent = 0.0
  for current in net_values:
    print(current)
    if current['High'] > peak:
      peak = current['High']
    drawdown = max(drawdown, peak - current['Low'])
    drawdown_percent = max(drawdown_percent, (peak - current['Low']) / peak)
  if print_summary:
    print(f"Max Drawdown: {drawdown: .2f}")
    print(f"Max % Drawdown: {drawdown_percent * 100: .2f}%")
  return {"Max Drawdown": drawdown, "Max % Drawdown": drawdown_percent}

def plot_net_values_chart(stock: str,
                          data: pd.DataFrame,
                          net_values: List[Dict[str, float]]):
  net_values_df = pd.DataFrame(net_values)
  net_values_df['Date'] = data['Date']
  net_values_df.set_index('Date', inplace = True)
  plotting.plot_lines(stock, net_values_df)
