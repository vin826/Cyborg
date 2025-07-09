import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


    
class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, window_size: int = 10, initial_balance: float = 1000.0):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.position_size = 0
        self.trade_log = []
        self.pnl_log = []
        self.fixed_risk_dollars = 10.0
        self.virtual_stop_pct = 0.02
        self.leverage = 5

        # For tracking drawdown
        self.current_trade_prices = []
        self.max_drawdown = 0.0
        self.unrealized_pnl = 0.0

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size * 8,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def _get_observation(self):
        window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        features = window[['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr', 'sentiment']].values
        return features.flatten()

    def step(self, action):
        price = self.data.iloc[self.current_step]["close"]
        reward = 0.0

        # --- Reward for opening a new trade ---
        if self.position == 0 and action in [1, 2]:
            reward += 0.1  # Encourage exploration

        # --- Reward/Penalty for closing a trade ---
        if self.position != 0 and (
            (self.position == 1 and action == 2) or (self.position == -1 and action == 1)
        ):
            pnl = (price - self.entry_price) * self.position
            reward += pnl * 0.01  # Scale profit/loss
            # Penalize large drawdown
            if self.max_drawdown > 0.01 * self.entry_price:
                reward -= self.max_drawdown * 0.005

        # --- Penalize holding a position too long ---
        if self.position != 0:
            reward -= 0.01  # Small penalty per step in a trade

        # --- Penalize holding (no position) ---
        if self.position == 0 and action == 0:
            reward -= 0.005  # Small penalty for doing nothing

        # --- Penalize every trade to discourage overtrading ---
        if action in [1, 2]:
            reward -= 0.002

        if self.position == 0:
            # No open trade → no floating PnL
            self.unrealized_pnl = 0.0

        # —————————————————————————————————————
        # 🔁 If in a position, update metrics & maybe close
        # —————————————————————————————————————
        if self.position != 0:
            # 1) Drawdown + floating PnL
            self.current_trade_prices.append(price)
            price_diff = (price - self.entry_price) * self.position
            self.unrealized_pnl = price_diff * self.position_size

            peak = max(self.current_trade_prices) if self.position == 1 else min(self.current_trade_prices)
            peak_pnl = (peak - self.entry_price) * self.position * self.position_size
            drawdown = peak_pnl - self.unrealized_pnl
            self.max_drawdown = max(self.max_drawdown, drawdown)

            # 2) Virtual stop check
            stop_price = (
                self.entry_price * (1 - self.virtual_stop_pct)
                if self.position == 1 else
                self.entry_price * (1 + self.virtual_stop_pct)
            )
            stop_breached = (self.position == 1 and price <= stop_price) \
                        or (self.position == -1 and price >= stop_price)

            # 3) 🔥 Take-Profit first
            tp_price = self.entry_price * (1 + 0.02 * self.position)
            if (self.position == 1 and price >= tp_price) \
            or (self.position == -1 and price <= tp_price):
                closing_reward = price_diff * self.position_size
                reward = closing_reward + 5.0     # +$5 bonus for hitting TP
                self.balance += closing_reward

                survivability_score = round(reward / (1 + self.max_drawdown), 4)
                self.trade_log.append({
                    "step": self.current_step,
                    "action": action,
                    "price": price,
                    "entry_price": self.entry_price,
                    "position": self.position,
                    "position_size": self.position_size,
                    "reward": closing_reward+5.0,
                    "risk_$": self.fixed_risk_dollars,
                    "stop_breached": stop_breached,
                    "max_drawdown": round(self.max_drawdown, 2),
                    "survivability_score": survivability_score,
                    "took_profit": True
                })

                # reset
                self.position = 0
                self.entry_price = 0
                self.position_size = 0
                self.current_trade_prices = []
                self.max_drawdown = 0.0
                self.unrealized_pnl = 0.0

            # 4) 🟥 Else, close on opposite action
            elif (self.position == 1 and action == 2) or (self.position == -1 and action == 1):
                trade_profit = price_diff * self.position_size
                reward += trade_profit
                self.balance += trade_profit

                survivability_score = round(reward / (1 + self.max_drawdown), 4)
                self.trade_log.append({
                    # same payload as above, but "took_profit": False
                    "step": self.current_step,
                    "action": action,
                    "price": price,
                    "entry_price": self.entry_price,
                    "position": self.position,
                    "position_size": self.position_size,
                    "reward": trade_profit,
                    "risk_$": self.fixed_risk_dollars,
                    "stop_breached": stop_breached,
                    "max_drawdown": round(self.max_drawdown, 2),
                    "survivability_score": survivability_score,
                    "took_profit": False
                })

                # reset
                self.position = 0
                self.entry_price = 0
                self.position_size = 0
                self.current_trade_prices = []
                self.max_drawdown = 0.0
                self.unrealized_pnl = 0.0

        # —————————————————————————————————————
        # 🟢 Open new position if none exists
        # —————————————————————————————————————
        if self.position == 0 and action in [1, 2]:
            self.position = 1 if action == 1 else -1
            self.entry_price = price

            stop_dist = price * self.virtual_stop_pct
            # your fixed-risk × leverage sizing
            units = self.fixed_risk_dollars / (stop_dist * self.leverage)
            self.position_size = units * self.leverage

            self.current_trade_prices = [price]
            self.max_drawdown = 0.0

            # Reward for opening a new trade
            reward += 0.1

        # Log the floating PnL each tick
        self.pnl_log.append({
            "step": self.current_step,
            "unrealized_pnl": self.unrealized_pnl,
            "position": self.position
        })

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, False, {}

    def print_observation(self):
        window = self.data.iloc[self.current_step - self.window_size:self.current_step]
        for i, row in window.iterrows():
            print(f"Step {i}:")
            print(f"  Close:     {row['close']:.2f}")
            print(f"  Volume:    {row['volume']:.2f}")
            print(f"  RSI:       {row['rsi']:.2f}")
            print(f"  MFI:       {row['mfi']:.2f}")
            print(f"  WT1 / WT2: {row['wt1']:.2f} / {row['wt2']:.2f}")
            print(f"  ATR:       {row['atr']:.2f}")
            print(f"  Sentiment: {row['sentiment']:.2f}")
            print("–––––")
    def get_trade_log(self):
        return pd.DataFrame(self.trade_log)

    def reset(self,seed=None, options=None):
        super().reset(seed=seed)  # This line is important for seeding
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.trade_log = []  # Clear previous trades
        self.pnl_log = []    # Clear previous P&L
        print(f"🔄 Environment reset. Starting balance: ${self.balance:.2f}")  # Debug line
        return self._get_observation(), {}  # Return observation and an empty info dict