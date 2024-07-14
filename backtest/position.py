from colorama import Fore, Style

class Position:
  def __init__(self,
               stock: str, date,
               price: float,
               shares: float):
    self.stock = stock
    self.open_date = date
    self.open_price = price
    self.shares = shares
    self.close_date = None
    self.close_price = None
    self.profit = None
    self.target_price = None
    self.stop_loss = None

  def close(self, date, price: float):
    self.close_date = date
    self.close_price = price
    self.profit = (self.close_price - self.open_price) * self.shares

  def is_closed(self):
    return self.close_date is not None

  def __eq__(self, other: "Position"):
    return self.stock == other.stock and self.open_date == other.open_date and self.open_price == other.open_price

  def print(self):
    if self.is_closed():
      percent_profit = (self.close_price - self.open_price) / self.open_price
      color = Fore.GREEN if self.profit > 0 else Fore.RED
      print(f"{color}{stock} Open Date: {self.open_date}, Price: {self.open_price: .2f}, Shares: {self.shares: .2f}, Close Date: {self.close_date}, Close Price: {self.close_price}, Profit: {self.profit}, % Profit: {percent_profit}\n{Style.RESET_ALL}")
    else:
      color = Fore.GRAY
      print(f"{color}{stock} Open Date: {self.open_date}, Price: {self.open_price: .2f}, Shares: {self.shares: .2f}\n{Style.RESET_ALL}")
