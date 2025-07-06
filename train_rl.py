from stable_baselines3 import PPO
from trading_env import TradingEnv
from ai_trader import AITrader
from utils.data_loader import load_ohlcv  # ✅ NEW
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

os.makedirs("models", exist_ok=True)

# Step 1: Load data from MEXC
df = load_ohlcv(
    symbol='BTC/USDT:USDT',
    timeframe='1h',
    limit=5000,
    api_key="mx0vglG2gFRGKGNGsd",
    api_secret="3fa8e335eb4b40c0a4fde65edcff401c"
)

# Step 2: Preprocess data with your AITrader logic
trader = AITrader(api_key="mx0vglG2gFRGKGNGsd", api_secret="3fa8e335eb4b40c0a4fde65edcff401c")
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

# Step 3: Build environment with enriched data
env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])  # Use DummyVecEnv for vectorized environment
eval_env = DummyVecEnv([lambda: Monitor(TradingEnv(df))])

# Create the callback for evaluation and model saving
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model",
    log_path="./logs/",
    eval_freq=5000,  # More frequent evaluation
    deterministic=True,
    render=False
)

# Step 4: Train your RL agent with improved hyperparameters and callback

# --- Encourage Exploration: Increase entropy, add hold penalty ---
def linear_schedule(initial_value):
    """
    Returns a function that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=linear_schedule(3e-4),
    verbose=1,
    normalize_advantage=True,
    gae_lambda=0.97,
    n_steps=4096,
    batch_size=128,
    ent_coef=0.05,  # Increased entropy for more exploration
    tensorboard_log="./tensorboard/"
)
model.learn(total_timesteps=1_500_000, callback=eval_callback)
model.save("models/sentiment_trader")

print("\nTraining complete! Model and logs saved.")

run_name = f"run_{int(time.time())}"
tensorboard_log = f"./tensorboard/{run_name}/"
print("\nTraining complete! Model and logs saved.")