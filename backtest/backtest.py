from trading.backtest.position import Position
from trading.backtest.trade import Trade

def backtest(stock,
             data,
             strategies,
             initial_capital = 100000,
             print_trades = False):
  data = data.copy()
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
  print_summary(data, initial_capital, cash, transactions, closed_positions)

def print_summary(data, initial_capital, final_capital, transactions, closed_positions):
  print(f"==={stock} Final Result===")
  total_return = final_capital - initial_capital
  percent_return = total_return / initial_capital
  # Annualize the return (assuming equal holding periods within a year)
  start_year = data['Date'][0].year
  end_year = data['Date'][len(data)-1].year
  holding_period = end_year - start_year + 1
  annualized_return = (1 + percent_return) ** (1 / holding_period) - 1
  # print(f"Return: {final_capital - initial_capital: .2f}")
  print(f"% Return: {percent_return * 100: .2f}%")
  print(f"Annualized Return: {annualized_return * 100: .2f}%")

  print_profit_trades(transactions)
  print_profit_factor(closed_positions)
  print_max_drawdown(initial_capital, transactions)

def print_profit_trades(transactions):
  trade_with_profit = 0
  trade_with_loss = 0
  # Print profit trades
  for trade in transactions:
    if trade.trade_type == 'Close':
      if trade.price > trade.unit_cost:
        trade_with_profit += 1
      else:
        trade_with_loss += 1
  print(f"Trade with Profit: {trade_with_profit: .2f}")
  print(f"Trade with loss: {trade_with_loss: .2f}")
  print(f"Profit Rate: {trade_with_profit / (trade_with_profit + trade_with_loss) * 100}%")

def print_profit_factor(closed_positions):
  gross_profit = 0
  gross_loss = 0
  for position in closed_positions:
    if position.profit > 0:
      gross_profit += position.profit
    else:
      gross_loss += position.profit
  print(f"Gross Profit: {gross_profit: .2f}")
  print(f"Gross Loss: {gross_loss: .2f}")
  print(f"Profit Factor: {gross_profit / -gross_loss}")

def print_max_drawdown(initial_capital, transactions):
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
  print(f"Max Drawdown: {drawdown: .2f}")
  print(f"Max % Drawdown: {drawdown_percent * 100: .2f}%")
