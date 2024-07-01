import termcolor
from colorama import Fore, Style

class Trade:
  def __init__(self,
               stock: str,
               date,
               price: float64,
               shares: float32,
               trade_type: str = "Long",
               unit_cost: float64 = 0):
    self.stock = stock
    self.date = date
    self.price = price
    self.shares = shares
    self.trade_type = trade_type
    self.unit_cost = unit_cost

  def print(self):
    if self.trade_type == 'Close':
      gain = (self.price - self.unit_cost) * self.shares
      percent_gain = (self.price - self.unit_cost) / self.unit_cost
      # color = 'green' if gain > 0 else 'red'
      # print(termcolor.colored(
      #     f"{self.trade_type} Date: {self.date}, Price: {self.price: .2f}, Shares: {self.shares: .2f}\nGain: {gain: .2f}, % Gain: {percent_gain*100}%\n",
      #     color=color))
      color = Fore.GREEN if gain > 0 else Fore.RED
      print(f"{color}{self.trade_type} {stock} Date: {self.date}, Price: {self.price: .2f}, Shares: {self.shares: .2f}\nGain: {gain: .2f}, % Gain: {percent_gain*100}%\n{Style.RESET_ALL}")
    else:
      print(f"{self.trade_type} {stock} Date: {self.date}, Price: {self.price: .2f}, Shares: {self.shares: .2f}\n")
