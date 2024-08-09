from trading.position import Position
from trading.trade import Trade
from trading import indicators

class Strategy:
  def maybe_open_position(self, stock, data, i, cash, positions):
    return []
  
  def maybe_close_position(self, stock, data, i, positions):
    return []

  def maybe_adjust_stop_win(self, stock, data, i, positions):
    return []

  def maybe_adjust_stop_loss(self, stock, data, i, positions):
    return []

  def generate_indicator_columns(self, data):
    return data

  def trade(self, stock, data, i, cash, positions):
    if len(positions) == 0:
      transactions = self.maybe_open_position(stock, data, i, cash, positions)
      if isinstance(transactions, Trade):
        transactions = [transactions]
    else:
      transactions = []

    stop_loss_trades = self._maybe_stop_loss(stock, data, i, positions)
    if isinstance(stop_loss_trades, Trade):
      transactions.append(stop_loss_trades)
    else:
      transactions.extend(stop_loss_trades)

    stop_win_trades = self._maybe_stop_win(stock, data, i, positions)
    if isinstance(stop_win_trades, Trade):
      transactions.append(stop_win_trades)
    else:
      transactions.extend(stop_win_trades)

    close_trades = self.maybe_close_position(stock, data, i, positions)
    if isinstance(close_trades, Trade):
      transactions.append(close_trades)
    else:
      transactions.extend(close_trades)

    if len(positions) > 0:
      self.maybe_adjust_stop_win(stock, data, i, positions)
      self.maybe_adjust_stop_loss(stock, data, i, positions)
    return transactions
  
  def _long(self, stock, data, i, cash, positions, date, price,
             target_price: float|None = None,
             stop_loss_price: float|None = None):
    shares = cash / price
    position = Position(stock, date, price, shares)
    position.target_price = target_price
    position.stop_loss_price = stop_loss_price
    positions.append(position)
    trade = Trade(stock,
                  date,
                  price,
                  shares,
                  trade_type = 'Long',
                  affected_position = position)
    return trade
  
  def _close(self, stock, data, i, positions, position_to_close, date, price,
              close_type: str = ""):
    trade = Trade(stock,
                  date,
                  price,
                  position_to_close.shares,
                  trade_type = f'Close - {close_type}',
                  unit_cost = position_to_close.open_price,
                  affected_position = position_to_close)
    position_to_close.close(date, price)
    positions.remove(position_to_close)
    return trade
  
  def _maybe_stop_win(self, stock, data, i, positions):
    date = data['Date'][i]
    transactions = []
    for position in positions:
      if position.target_price is not None and data['High'][i] >= position.target_price:
        price = position.target_price
        transactions.append(self._close(stock, data, i, positions, position, date, price, close_type = 'stop win'))
    return transactions
  
  def _maybe_stop_loss(self, stock, data, i, positions):
    date = data['Date'][i]
    transactions = []
    for position in positions:
      if position.stop_loss_price is not None and data['Low'][i] <= position.stop_loss_price:
        price = min(position.stop_loss_price, data['Open'][i])
        transactions.append(self._close(stock, data, i, positions, position, date, price, close_type = 'stop loss'))
    return transactions
