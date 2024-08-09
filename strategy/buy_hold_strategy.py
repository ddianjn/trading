from trading.position import Position
from trading.trade import Trade
from trading.strategy.strategy import Strategy
from trading import indicators

class BuyHoldStrategy(Strategy):
  def maybe_open_position(self, stock, data, i, cash, positions):
    if i == 0:
      return self._long(stock, data, i, cash, positions,
                        data['Date'][i],
                        data['Open'][i])
    else:
      return []
