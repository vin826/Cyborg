import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv
from ai_trader import AITrader
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from utils.data_loader import load_ohlcv  # ✅ NEW
from matplotlib.animation import FuncAnimation, PillowWriter



def compute_risk(df, stop_pct=0.01, fixed_risk=10.0):
    df = df.copy()
    # Use 'reward' if present, else try 'pnl', else raise error
    reward_col = None
    for col in ['reward', 'pnl', 'profit', 'net_reward']:
        if col in df.columns:
            reward_col = col
            break
    if reward_col is None:
        raise KeyError("No reward/profit column found in trade log. Columns: " + str(df.columns))
    df["risk_$"] = fixed_risk
    assert (df['risk_$'] <= fixed_risk + 1e-6).all()
    df["potential_gain"] = df[reward_col]
    df["reward_to_risk"] = df[reward_col] / df["risk_$"]
    return df




def plot_trade_replay(df, price_series):
    plt.figure(figsize=(14, 6))
    plt.plot(price_series.index, price_series.values, label="Price", alpha=0.6)

    for i, trade in df.iterrows():
        color = "green" if trade["action"] == 1 else "red"
        label = f'{["Hold", "Buy", "Sell"][trade["action"]]} @ {trade["price"]:.0f}'
        plt.scatter(trade["step"], trade["price"], c=color, s=50, label=label, alpha=0.7)

    plt.title("📉 Cyborg Replay: Trades Over Price")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

def animate_agent_windows(env, save_path="cyborg_observation.gif"):
    fig, ax = plt.subplots(figsize=(10, 4))
    obs_matrix = env.data[['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr', 'sentiment']].values

    def update(frame):
        ax.clear()
        window = obs_matrix[frame - env.window_size:frame].T
        im = ax.imshow(window, aspect='auto', cmap='coolwarm')
        ax.set_yticks(range(window.shape[0]))
        ax.set_yticklabels(['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr', 'sentiment'])
        ax.set_xticks(range(env.window_size))
        ax.set_xticklabels([f"t-{i}" for i in range(env.window_size-1, -1, -1)])
        ax.set_title(f"Cyborg Perception — t={frame}")
        return [im]

    anim = FuncAnimation(fig, update, frames=range(env.window_size, len(obs_matrix)), blit=True)
    anim.save(save_path, writer=PillowWriter(fps=2))
    plt.close()
    print(f"🎬 Animation saved to: {save_path}")
def plot_observation_window(env):
    # Extract observation matrix for the current step
    window_data = env.data[['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr', 'sentiment']].iloc[
        env.current_step - env.window_size : env.current_step
    ].T

    plt.figure(figsize=(10, 4))
    plt.imshow(window_data.values, aspect='auto', cmap='coolwarm', interpolation='nearest')

    plt.yticks(range(window_data.shape[0]), window_data.index)
    plt.xticks(range(env.window_size), [f"t-{i}" for i in range(env.window_size - 1, -1, -1)])
    plt.colorbar(label='Feature Value')
    plt.title("🤖 Cyborg Input Window — What the Model Sees")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()
# Load model
model = PPO.load("models/sentiment_trader")

# Fetch fresh data
trader = AITrader(api_key="mx0vglG2gFRGKGNGsd", api_secret="3fa8e335eb4b40c0a4fde65edcff401c")
df = trader.fetch_data(symbol="BTC/USDT:USDT", timeframe="1h", limit=1000)

print("\n=== Raw Data ===")
print(f"Shape: {df.shape}")
print(f"Columns: {', '.join(df.columns)}")
print("\nFirst few rows:")
print(df.head().to_string(index=False))

df = trader.preprocess(df)

# === Feature Diagnostics ===

feature_cols = ['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr', 'sentiment']

# Plot time series for each feature
df[feature_cols].plot(subplots=True, figsize=(12, 16), title="Feature Time Series")
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nFeature summary statistics:")
print(df[feature_cols].describe())

# Check for NaNs or Infs
print("Any NaNs in features?", df[feature_cols].isna().any())
print("Any Infs in features?", np.isinf(df[feature_cols].values).any())

# Check feature correlations
print("\nFeature correlation matrix:")
print(df[feature_cols].corr())
# === End Feature Diagnostics ===

# Initialize environment
env = TradingEnv(df)
obs, _ = env.reset()
done = False
while not done:
    action = np.random.choice([0, 1, 2])
    obs, reward, done, truncated, _ = env.step(action)
print("Random agent trade log length:", len(env.get_trade_log()))

# --- Improved Inference and Reporting Loop ---
actions = ["HOLD", "BUY", "SELL"]
obs, _ = env.reset()
done = False
total_reward = 0
step_rewards = []
action_history = []
confidence_history = []

print("\n=== Agent Trading Simulation ===")
while not done:
    # Get action probabilities (logits to softmax)
    obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    logits = model.policy.forward(obs_tensor)[0].float()
    probs = torch.nn.functional.softmax(logits, dim=-1).detach().numpy().flatten()
    action = np.argmax(probs)
    confidence = probs[action]
    action_history.append(action)
    confidence_history.append(confidence)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    step_rewards.append(reward)
    print(f"Step {env.current_step:4d} | 🧠 Action: {actions[action]:4s} | Confidence: {confidence:.2%} | Reward: {reward:.2f} | Balance: ${env.balance:.2f}")

print("\n=== Trading Results ===")
print(f"Final balance: ${env.balance:.2f}")
print(f"Total reward: {total_reward:.2f}")
print(f"Final position: {env.position} (0 = neutral, 1 = long, -1 = short)")
print(f"Final entry price: ${env.entry_price:.2f}")
print(f"Trading duration: {env.current_step} steps")

print("\n=== Final Observation ===")
env.print_observation()

# Show last observation window
try:
    obs_reshaped = obs.reshape(-1, 10)
    for i, row in enumerate(obs_reshaped):
        print(f"Window {i+1}: {row}")
except Exception as e:
    print("Could not reshape observation for display:", e)

# Trade log and risk analysis
log = env.get_trade_log()
print("Trade log columns:", log.columns)

print(log.head())

if log.empty:
    print("No trades were made. Skipping risk analysis.")
else:
    log = compute_risk(log)
    #plot_trade_replay(log, env.data['close'])
    risked_trades = compute_risk(log)
    print(risked_trades[["step", "action", "entry_price", "reward", "risk_$", "reward_to_risk"]])
    print("🔢 Total Trades:", len(risked_trades))
    print("💸 Avg Risk per Trade: $", round(risked_trades["risk_$"].mean(), 2))
    print("📊 Avg Reward:", round(risked_trades["reward"].mean(), 2))
    print("🏆 Avg Reward-to-Risk:", round(risked_trades["reward_to_risk"].mean(), 2))

# Top survivors
df = pd.DataFrame(env.trade_log)
if not df.empty and "survivability_score" in df.columns:
    top_survivors = df.sort_values("survivability_score", ascending=False).head(10)
    print(top_survivors[["step", "reward", "max_drawdown", "survivability_score"]])

print(f"Total steps: {env.current_step}")
print(f"Final balance: {env.balance}")
print(f"Trade log length: {len(env.trade_log)}")

# Plot confidence and actions over time
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(confidence_history, label="Action Confidence")
    plt.title("Agent Action Confidence Over Time")
    plt.xlabel("Step")
    plt.ylabel("Confidence")
    plt.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Could not plot confidence history:", e)

print("\n=== Observation Window Plot ===")
plot_observation_window(env)

#print("\n=== Animation of Agent's Windows ===")
#animate_agent_windows(env)
print("Action counts:", {actions[a]: action_history.count(a) for a in set(action_history)})

if not log.empty and "took_profit" in log.columns:
    took_profit_trades = log[log["took_profit"] == True]
    print("\n=== Trades Where Agent Took Profit ===")
    if not took_profit_trades.empty:
        print(took_profit_trades[["step", "action", "entry_price", "price", "reward", "took_profit"]].to_string(index=False))
    else:
        print("No trades where agent took profit.")

if not log.empty and "reward" in log.columns:
    print("\n=== Trades Where Agent Took Profit (reward > 0) ===")
    took_profit_trades = log[log["reward"] > 0]
    if not took_profit_trades.empty:
        print(took_profit_trades[["step", "action", "entry_price", "price", "reward"]].to_string(index=False))
    else:
        print("No trades where agent took profit.")

    print("\n=== Trades Where Agent Took a Loss (reward < 0) ===")
    took_loss_trades = log[log["reward"] < 0]
    if not took_loss_trades.empty:
        print(took_loss_trades[["step", "action", "entry_price", "price", "reward"]].to_string(index=False))
    else:
        print("No trades where agent took a loss.")
