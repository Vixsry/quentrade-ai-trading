#!/usr/bin/env python3
"""
Quentrade Utility Functions
Helper functions and utilities for Quentrade
"""

import os
import json
import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from cryptography.fernet import Fernet
import yaml
import csv

class QuentradeUtils:
    """Utility functions for Quentrade"""
    
    @staticmethod
    def encrypt_api_key(api_key: str, encryption_key: bytes = None) -> Tuple[str, bytes]:
        """Encrypt API key for secure storage"""
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        fernet = Fernet(encryption_key)
        encrypted_key = fernet.encrypt(api_key.encode())
        
        return encrypted_key.decode(), encryption_key
    
    @staticmethod
    def decrypt_api_key(encrypted_key: str, encryption_key: bytes) -> str:
        """Decrypt API key"""
        fernet = Fernet(encryption_key)
        decrypted_key = fernet.decrypt(encrypted_key.encode())
        
        return decrypted_key.decode()
    
    @staticmethod
    def generate_bybit_signature(api_secret: str, params: Dict) -> str:
        """Generate signature for Bybit API requests"""
        param_str = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(api_secret.encode('utf-8'), param_str.encode('utf-8'), hashlib.sha256).hexdigest()
    
    @staticmethod
    def format_timeframe(timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        timeframe_map = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '12h': 720,
            '1d': 1440,
            '1w': 10080,
            '1M': 43200
        }
        
        return timeframe_map.get(timeframe, 60)
    
    @staticmethod
    def calculate_position_size(capital: float, risk_percent: float, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_amount = capital * risk_percent
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return 0
            
        position_size = risk_amount / price_difference
        return position_size
    
    @staticmethod
    def calculate_liquidation_price(entry_price: float, leverage: int, side: str) -> float:
        """Calculate liquidation price for a position"""
        if side.upper() == 'LONG':
            liquidation_price = entry_price * (1 - 1/leverage + 0.005)  # 0.5% maintenance margin
        else:
            liquidation_price = entry_price * (1 + 1/leverage - 0.005)
            
        return liquidation_price
    
    @staticmethod
    def format_large_number(number: float) -> str:
        """Format large numbers for better readability"""
        if number >= 1_000_000_000:
            return f"{number/1_000_000_000:.2f}B"
        elif number >= 1_000_000:
            return f"{number/1_000_000:.2f}M"
        elif number >= 1_000:
            return f"{number/1_000:.2f}K"
        else:
            return f"{number:.2f}"
    
    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 14) -> float:
        """Calculate historical volatility"""
        if len(prices) < period:
            return 0
            
        returns = np.diff(np.log(prices))
        volatility = np.std(returns[-period:]) * np.sqrt(365)  # Annualized
        
        return volatility
    
    @staticmethod
    def detect_market_regime(prices: pd.Series, volume: pd.Series) -> str:
        """Detect current market regime"""
        # Simple regime detection based on volatility and trend
        volatility = prices.pct_change().rolling(20).std().iloc[-1]
        trend = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20]
        volume_trend = (volume.iloc[-1] - volume.mean()) / volume.std()
        
        if volatility > 0.03:  # High volatility threshold
            if abs(trend) > 0.1:  # Strong trend
                return "TRENDING_VOLATILE"
            else:
                return "RANGING_VOLATILE"
        else:
            if abs(trend) > 0.05:  # Moderate trend
                return "TRENDING_STABLE"
            else:
                return "RANGING_STABLE"
    
    @staticmethod
    def export_signals_to_csv(signals: List[Dict], filename: str):
        """Export trading signals to CSV file"""
        if not signals:
            return
            
        keys = signals[0].keys()
        
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(signals)
    
    @staticmethod
    def load_strategy_from_yaml(filepath: str) -> Dict:
        """Load trading strategy from YAML file"""
        with open(filepath, 'r') as file:
            strategy = yaml.safe_load(file)
        return strategy
    
    @staticmethod
    def save_strategy_to_yaml(strategy: Dict, filepath: str):
        """Save trading strategy to YAML file"""
        with open(filepath, 'w') as file:
            yaml.dump(strategy, file, default_flow_style=False)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        
        if len(excess_returns) < 2 or excess_returns.std() == 0:
            return 0
            
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2 or downside_returns.std() == 0:
            return 0
            
        sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()
    
    @staticmethod
    def normalize_symbol(symbol: str, exchange: str = 'bybit') -> str:
        """Normalize trading pair symbols across different exchanges"""
        symbol = symbol.upper().replace('/', '').replace('-', '')
        
        if exchange.lower() == 'binance':
            # Binance uses direct concatenation: BTCUSDT
            return symbol
        elif exchange.lower() == 'bybit':
            # Bybit uses direct concatenation: BTCUSDT
            return symbol
        elif exchange.lower() == 'coinbase':
            # Coinbase uses hyphen: BTC-USD
            if 'USDT' in symbol:
                return symbol.replace('USDT', '-USDT')
            elif 'USD' in symbol:
                return symbol.replace('USD', '-USD')
            return symbol
        else:
            return symbol
    
    @staticmethod
    def check_trading_hours(timezone: str = 'UTC') -> bool:
        """Check if current time is within trading hours"""
        # Crypto markets are 24/7, but you might want to avoid certain hours
        current_hour = datetime.now().hour
        
        # Example: avoid trading during low liquidity hours (optional)
        if 2 <= current_hour <= 6:  # 2 AM to 6 AM
            return False
            
        return True
    
    @staticmethod
    def format_trade_summary(trade_data: Dict) -> str:
        """Format trade data into a readable summary"""
        summary = f"""
Trade Summary
=============
Symbol: {trade_data.get('symbol', 'N/A')}
Side: {trade_data.get('side', 'N/A')}
Entry Price: ${trade_data.get('entry_price', 0):.2f}
Exit Price: ${trade_data.get('exit_price', 0):.2f}
Quantity: {trade_data.get('quantity', 0):.4f}
P&L: ${trade_data.get('pnl', 0):.2f}
ROI: {trade_data.get('roi', 0):.2f}%
Duration: {trade_data.get('duration', 'N/A')}
Result: {trade_data.get('result', 'N/A')}
"""
        return summary
    
    @staticmethod
    def validate_api_response(response: Dict) -> bool:
        """Validate API response structure"""
        if not isinstance(response, dict):
            return False
            
        # Check for common error indicators
        error_keys = ['error', 'err_code', 'ret_code']
        for key in error_keys:
            if key in response and response[key] not in [0, '0', None, '']:
                return False
                
        return True
    
    @staticmethod
    def retry_with_backoff(func, max_retries: int = 3, initial_delay: float = 1.0):
        """Retry function with exponential backoff"""
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                    
                time.sleep(delay)
                delay *= 2  # Exponential backoff
                
        return None