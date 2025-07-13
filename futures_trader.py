import ccxt
import time
import pandas as pd
from stable_baselines3 import PPO
from ai_trader import AITrader
import numpy as np
from datetime import datetime

class FuturesLiveTradingAgent:
    def __init__(self, model_path="models/sentiment_trader"):
        # Load your trained model
        self.model = PPO.load(model_path)
        
        # Initialize MEXC exchange for futures
        self.exchange = ccxt.mexc({
            'apiKey': "mx0vglG2gFRGKGNGsd",
            'secret': "3fa8e335eb4b40c0a4fde65edcff401c",
            'sandbox': False,  # Set to True for testing
            'options': {
                'defaultType': 'swap',  # This enables futures trading
            }
        })
        
        # Trading state
        self.position_size = 0  # Current position size
        self.entry_price = 0
        self.trade_log = []
        self.window_size = 10
        
        print(f"🚀 Futures Trading Agent Initialized!")
        self.check_account_status()
        
    def check_account_status(self):
        """Check your actual futures account balance and status"""
        try:
            # Get futures balance
            balance = self.exchange.fetch_balance()
            
            print("\n💰 FUTURES ACCOUNT STATUS:")
            print("="*40)
            
            # Show USDT balance (main futures currency)
            if 'USDT' in balance:
                usdt_balance = balance['USDT']
                print(f"💵 USDT Balance: {usdt_balance['total']:.2f}")
                print(f"💵 Available: {usdt_balance['free']:.2f}")
                print(f"💵 Used (in positions): {usdt_balance['used']:.2f}")
            
            # Check current positions
            positions = self.exchange.fetch_positions()
            active_positions = [pos for pos in positions if float(pos['contracts']) != 0]
            
            if active_positions:
                print(f"\n📊 ACTIVE POSITIONS: {len(active_positions)}")
                for pos in active_positions:
                    print(f"  {pos['symbol']}: {pos['contracts']} contracts")
                    print(f"    Entry: ${pos['entryPrice']}")
                    print(f"    PnL: ${pos['unrealizedPnl']:.2f}")
            else:
                print("\n📊 No active positions")
                
            return balance
            
        except Exception as e:
            print(f"❌ Error checking account: {e}")
            return None
    
    def get_futures_balance(self):
        """Get current USDT futures balance"""
        try:
            balance = self.exchange.fetch_balance()
            if 'USDT' in balance:
                return balance['USDT']['free']  # Available balance
            return 0
        except Exception as e:
            print(f"❌ Error getting balance: {e}")
            return 0
    
    def get_current_position(self, symbol='BTC/USDT:USDT'):
        """Get current position for BTC futures"""
        try:
            positions = self.exchange.fetch_positions([symbol])
            if positions:
                pos = positions[0]
                return {
                    'size': float(pos['contracts']),
                    'side': pos['side'],  # 'long' or 'short'
                    'entry_price': float(pos['entryPrice']) if pos['entryPrice'] else 0,
                    'unrealized_pnl': float(pos['unrealizedPnl']) if pos['unrealizedPnl'] else 0
                }
            return {'size': 0, 'side': None, 'entry_price': 0, 'unrealized_pnl': 0}
        except Exception as e:
            print(f"❌ Error getting position: {e}")
            return {'size': 0, 'side': None, 'entry_price': 0, 'unrealized_pnl': 0}
    
    def place_futures_order(self, action, current_price, position_size_usd=100):
        """Place futures buy/sell orders"""
        symbol = 'BTC/USDT:USDT'
        
        try:
            # Get current position
            current_pos = self.get_current_position(symbol)
            
            # Calculate position size in BTC
            btc_amount = position_size_usd / current_price
            
            if action == 1:  # BUY (Long)
                if current_pos['size'] == 0:  # No current position
                    print(f"🟢 OPENING LONG POSITION")
                    print(f"   Size: ${position_size_usd} ({btc_amount:.6f} BTC)")
                    print(f"   Price: ${current_price:.2f}")
                    
                    # Place market buy order for futures
                    order = self.exchange.create_market_buy_order(
                        symbol=symbol,
                        amount=btc_amount,
                        params={'type': 'swap'}  # Futures order
                    )
                    
                    print(f"✅ Long order placed: {order['id']}")
                    
                    self.trade_log.append({
                        'timestamp': datetime.now(),
                        'action': 'LONG',
                        'price': current_price,
                        'size': btc_amount,
                        'order_id': order['id']
                    })
                    
                    return True
                    
            elif action == 2:  # SELL (Close Long or Open Short)
                if current_pos['size'] > 0:  # Close long position
                    print(f"🔴 CLOSING LONG POSITION")
                    print(f"   Size: {current_pos['size']:.6f} BTC")
                    print(f"   Entry: ${current_pos['entry_price']:.2f}")
                    print(f"   Current: ${current_price:.2f}")
                    
                    # Calculate P&L
                    pnl_percent = ((current_price - current_pos['entry_price']) / current_pos['entry_price']) * 100
                    print(f"   P&L: {pnl_percent:.2f}%")
                    
                    # Close the position
                    order = self.exchange.create_market_sell_order(
                        symbol=symbol,
                        amount=current_pos['size'],
                        params={'type': 'swap', 'reduceOnly': True}
                    )
                    
                    print(f"✅ Position closed: {order['id']}")
                    
                    self.trade_log.append({
                        'timestamp': datetime.now(),
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'entry_price': current_pos['entry_price'],
                        'pnl_percent': pnl_percent,
                        'order_id': order['id']
                    })
                    
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error placing futures order: {e}")
            return False
    
    def get_current_observation(self):
        """Get current market data for the AI model"""
        try:
            # Get recent data
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT:USDT', '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if df.empty:
                return None
                
            # Use AITrader to preprocess (add indicators)
            trader = AITrader(api_key="mx0vglG2gFRGKGNGsd", api_secret="3fa8e335eb4b40c0a4fde65edcff401c")
            df = trader.preprocess(df)
            
            if len(df) < self.window_size:
                return None
            
            # Get observation window
            feature_cols = ['close', 'volume', 'rsi', 'mfi', 'wt1', 'wt2', 'atr', 'sentiment']
            window_data = df[feature_cols].tail(self.window_size).values
            observation = window_data.flatten()
            
            return observation, df['close'].iloc[-1]
            
        except Exception as e:
            print(f"❌ Error getting observation: {e}")
            return None
    
    def start_futures_trading(self, check_interval=300, position_size_usd=100):
        """Start live futures trading"""
        print(f"🚀 Starting Futures Trading!")
        print(f"💰 Position Size: ${position_size_usd} per trade")
        print(f"⏰ Check Interval: {check_interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Check account status
                balance = self.get_futures_balance()
                current_pos = self.get_current_position()
                
                print(f"💰 Available Balance: ${balance:.2f}")
                if current_pos['size'] != 0:
                    print(f"📊 Current Position: {current_pos['side']} {current_pos['size']:.6f} BTC")
                    print(f"📈 Unrealized P&L: ${current_pos['unrealized_pnl']:.2f}")
                
                # Get AI prediction
                obs_data = self.get_current_observation()
                if obs_data is None:
                    print("⚠️ No market data, skipping...")
                    time.sleep(check_interval)
                    continue
                
                observation, current_price = obs_data
                action, _states = self.model.predict(observation, deterministic=True)
                action = int(action)
                
                actions = ["HOLD", "BUY", "SELL"]
                print(f"🧠 AI Decision: {actions[action]} | BTC Price: ${current_price:.2f}")
                
                # Execute trade
                if action != 0 and balance > position_size_usd:  # Not HOLD and have enough balance
                    success = self.place_futures_order(action, current_price, position_size_usd)
                    if success:
                        print("✅ Futures trade executed!")
                    else:
                        print("❌ Trade failed")
                else:
                    if balance <= position_size_usd:
                        print("⚠️ Insufficient balance for new position")
                    else:
                        print("⏸️ Holding...")
                
                # Wait for next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Futures trading stopped")
            self.show_trading_summary()
    
    def show_trading_summary(self):
        """Show trading performance summary"""
        print("\n" + "="*50)
        print("📊 FUTURES TRADING SUMMARY")
        print("="*50)
        
        # Final balance check
        final_balance = self.get_futures_balance()
        print(f"💰 Final Balance: ${final_balance:.2f}")
        
        # Show trades
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
            print(f"📈 Total Trades: {len(df)}")
            
            for trade in self.trade_log:
                print(f"\n{trade['timestamp'].strftime('%H:%M:%S')} - {trade['action']}")
                print(f"  Price: ${trade['price']:.2f}")
                if 'pnl_percent' in trade:
                    print(f"  P&L: {trade['pnl_percent']:.2f}%")

# Start futures trading
if __name__ == "__main__":
    # Create futures trading agent
    futures_agent = FuturesLiveTradingAgent(model_path="models/sentiment_trader")
    
    # Start with small position size for safety!
    futures_agent.start_futures_trading(
        check_interval=300,  # Check every 5 minutes
        position_size_usd=50  # Start with $50 positions
    )