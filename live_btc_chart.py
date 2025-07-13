import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import pandas as pd
from ai_trader import AITrader
import numpy as np

class LiveBTCChart:
    def __init__(self):
        self.trader = AITrader(
            api_key="mx0vglG2gFRGKGNGsd", 
            api_secret="3fa8e335eb4b40c0a4fde65edcff401c"
        )
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        self.prices = []
        self.volumes = []
        self.timestamps = []
        
    def fetch_latest_data(self):
        """Get the latest BTC data"""
        try:
            # Get last 50 candles for context
            df = self.trader.fetch_data(symbol="BTC/USDT:USDT", timeframe="1h", limit=50)
            if not df.empty:
                return df
        except Exception as e:
            print(f"Error fetching data: {e}")
        return None
    
    def update_chart(self, frame):
        """Update the chart with new data"""
        df = self.fetch_latest_data()
        
        if df is not None and not df.empty:
            # Clear the axes
            self.ax1.clear()
            self.ax2.clear()
            
            # Plot price line
            self.ax1.plot(range(len(df)), df['close'], color='orange', linewidth=2, label='BTC/USDT')
            self.ax1.set_title(f'🚀 LIVE BTC/USDT - ${df["close"].iloc[-1]:.2f} - {datetime.now().strftime("%H:%M:%S")}', 
                              fontsize=16, color='orange')
            self.ax1.set_ylabel('Price ($)', fontsize=12)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            
            # Add some technical indicators
            if len(df) > 20:
                # Simple moving averages
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_10'] = df['close'].rolling(10).mean()
                
                self.ax1.plot(range(len(df)), df['sma_20'], color='blue', alpha=0.7, label='SMA 20')
                self.ax1.plot(range(len(df)), df['sma_10'], color='red', alpha=0.7, label='SMA 10')
                self.ax1.legend()
            
            # Plot volume
            colors = ['green' if df.iloc[i]['close'] >= df.iloc[i]['open'] else 'red' 
                     for i in range(len(df))]
            self.ax2.bar(range(len(df)), df['volume'], color=colors, alpha=0.6)
            self.ax2.set_title('📊 Volume', fontsize=14)
            self.ax2.set_xlabel('Time (Hours Ago)', fontsize=12)
            self.ax2.set_ylabel('Volume', fontsize=12)
            self.ax2.grid(True, alpha=0.3)
            
            # Format x-axis to show time
            if len(df) > 10:
                step = max(1, len(df) // 10)
                self.ax1.set_xticks(range(0, len(df), step))
                self.ax1.set_xticklabels([f"-{len(df)-i}h" for i in range(0, len(df), step)])
                self.ax2.set_xticks(range(0, len(df), step))
                self.ax2.set_xticklabels([f"-{len(df)-i}h" for i in range(0, len(df), step)])
            
            # Add current price annotation
            current_price = df['close'].iloc[-1]
            self.ax1.annotate(f'${current_price:.2f}', 
                            xy=(len(df)-1, current_price), 
                            xytext=(len(df)-1, current_price),
                            fontsize=12, fontweight='bold', color='orange',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
    
    def start_live_chart(self):
        """Start the live updating chart"""
        print("🚀 Starting Live BTC Chart... Press Ctrl+C to stop")
        
        # Update every 60 seconds (1 minute)
        ani = animation.FuncAnimation(self.fig, self.update_chart, interval=60000, cache_frame_data=False)
        
        plt.show()
        return ani

# Create and start the live chart
if __name__ == "__main__":
    live_chart = LiveBTCChart()
    animation_obj = live_chart.start_live_chart()