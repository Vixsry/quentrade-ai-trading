#!/usr/bin/env python3
"""
Quentrade - AI Machine Learning Trading Terminal Crypto
Professional AI Trading Assistant for Bybit Exchange
"""

import os
import sys
import asyncio
import json
import time
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import configparser
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple, Optional
import aiohttp
import websockets
import feedparser
from textblob import TextBlob
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import hashlib
import hmac

# Initialize console
console = Console()

class QuentradeAI:
    def __init__(self):
        self.console = console
        self.version = "1.0.0"
        self.config_file = "config.ini"
        self.db_file = "quentrade.db"
        self.models_dir = "models"
        self.logs_dir = "logs"
        self.signals_history = []
        self.strategies = []
        self.current_model = None
        self.api_keys = {}
        self.initialize_directories()
        self.initialize_database()
        self.load_api_keys()
        self.load_models()
        
    def initialize_directories(self):
        """Create necessary directories"""
        dirs = [self.models_dir, self.logs_dir]
        for dir_path in dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    
    def initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                coin TEXT,
                direction TEXT,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                confidence REAL,
                reasoning TEXT,
                status TEXT,
                result REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                source TEXT,
                headline TEXT,
                sentiment TEXT,
                impact_score REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                rules TEXT,
                created_at DATETIME,
                performance_score REAL,
                active BOOLEAN
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_api_keys(self):
        """Load API keys from .env file"""
        load_dotenv()
        self.api_keys = {
            'BYBIT_API_KEY': os.getenv('BYBIT_API_KEY'),
            'BYBIT_API_SECRET': os.getenv('BYBIT_API_SECRET'),
            'COINMARKETCAP_API_KEY': os.getenv('COINMARKETCAP_API_KEY'),
            'FRED_API_KEY': os.getenv('FRED_API_KEY'),
            'NEWS_API_KEY': os.getenv('NEWS_API_KEY'),
            'QUIKNODE_API_KEY': os.getenv('QUIKNODE_API_KEY')
        }
    
    def load_models(self):
        """Load AI models"""
        try:
            if os.path.exists(f"{self.models_dir}/main_model.h5"):
                self.current_model = tf.keras.models.load_model(f"{self.models_dir}/main_model.h5")
            else:
                self.create_new_model()
        except Exception as e:
            self.console.print(f"[red]Error loading models: {str(e)}[/red]")
            self.create_new_model()
    
    def create_new_model(self):
        """Create new AI model"""
        # Simple LSTM model for demonstration
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(30, 15), return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # 3 outputs: UP, DOWN, NEUTRAL
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.current_model = model
        model.save(f"{self.models_dir}/main_model.h5")
    
    def show_banner(self):
        """Display Quentrade banner"""
        banner = """
[bold cyan]
 ████████╗ ██╗   ██╗███████╗███╗   ██╗████████╗██████╗  █████╗ ██████╗ ███████╗
 ██╔══██╗██║   ██║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝
 ██║  ██║██║   ██║█████╗  ██╔██╗ ██║   ██║   ██████╔╝███████║██║  ██║█████╗  
 ██║  ██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  
 ██████╔╝╚██████╔╝███████╗██║ ╚████║   ██║   ██║  ██║██║  ██║██████╔╝███████╗
 ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝
[/bold cyan]
[bold yellow]AI-Powered Crypto Trading Terminal - Professional Edition v{self.version}[/bold yellow]
        """
        self.console.print(banner)
    
    def show_main_menu(self):
        """Display main menu"""
        menu_items = [
            "[1] SCAN COIN - Analyze specific coin",
            "[2] NEWS SCAN - Latest crypto & economic news",
            "[3] CPI DATA - Economic indicators analysis",
            "[4] SIGNAL TRADING - AI trading signals",
            "[5] STRATEGY STUDIO - Create & manage strategies",
            "[6] BACKTEST ENGINE - Test historical performance",
            "[7] SETTINGS - Configuration & API keys",
            "[8] LOG & STATS - Performance statistics",
            "[9] EXIT - Close application"
        ]
        
        panel = Panel("\n".join(menu_items), title="[bold green]MAIN MENU[/bold green]", border_style="green")
        self.console.print(panel)
    
    async def scan_coin(self):
        """Scan and analyze specific coin"""
        self.console.print("[bold green]COIN SCANNER[/bold green]")
        coin = prompt("Enter coin pair (e.g., BTCUSDT): ").upper()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Analyzing {coin}...", total=5)
            
            # Collect market data
            progress.update(task, advance=1, description="[cyan]Collecting market data...")
            market_data = await self.get_market_data(coin)
            
            # Collect news data
            progress.update(task, advance=1, description="[cyan]Analyzing news sentiment...")
            news_sentiment = await self.analyze_news_sentiment(coin)
            
            # Collect on-chain data
            progress.update(task, advance=1, description="[cyan]Fetching on-chain data...")
            on_chain_data = await self.get_on_chain_data(coin)
            
            # Collect economic data
            progress.update(task, advance=1, description="[cyan]Checking economic indicators...")
            economic_data = await self.get_economic_data()
            
            # AI analysis
            progress.update(task, advance=1, description="[cyan]Running AI analysis...")
            analysis = self.ai_analyze_coin(coin, market_data, news_sentiment, on_chain_data, economic_data)
        
        # Display results
        table = Table(title=f"Analysis Results for {coin}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Direction", analysis['direction'])
        table.add_row("Entry Price", f"${analysis['entry_price']:.2f}")
        table.add_row("Stop Loss", f"${analysis['stop_loss']:.2f}")
        table.add_row("Take Profit", f"${analysis['take_profit']:.2f}")
        table.add_row("Confidence", f"{analysis['confidence']:.1f}%")
        table.add_row("Risk/Reward", f"1:{analysis['risk_reward']:.1f}")
        
        self.console.print(table)
        self.console.print(f"\n[bold yellow]AI Reasoning:[/bold yellow] {analysis['reasoning']}")
    
    async def news_scan(self):
        """Scan and analyze news"""
        self.console.print("[bold green]NEWS SCANNER[/bold green]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Scanning news sources...", total=15)
            
            news_sources = [
                "https://cryptopanic.com/news/",
                "https://cointelegraph.com/rss",
                "https://bitcoinist.com/feed/",
                "https://cryptopotato.com/feed/",
                # Add more sources...
            ]
            
            all_news = []
            for source in news_sources:
                try:
                    feed = feedparser.parse(source)
                    for entry in feed.entries[:5]:  # Get top 5 from each source
                        sentiment = self.analyze_sentiment(entry.title)
                        all_news.append({
                            'headline': entry.title,
                            'source': feed.feed.title,
                            'timestamp': entry.published if hasattr(entry, 'published') else datetime.now(),
                            'sentiment': sentiment,
                            'impact': self.calculate_impact_score(entry.title, sentiment)
                        })
                    progress.advance(task)
                except Exception as e:
                    self.console.print(f"[red]Error fetching from {source}: {str(e)}[/red]")
        
        # Display news
        table = Table(title="Latest Crypto News", show_lines=True)
        table.add_column("Time", style="cyan")
        table.add_column("Headline", style="white")
        table.add_column("Sentiment", style="green")
        table.add_column("Impact", style="yellow")
        
        for news in sorted(all_news, key=lambda x: x['impact'], reverse=True)[:20]:
            sentiment_color = "green" if news['sentiment'] > 0 else "red" if news['sentiment'] < 0 else "yellow"
            table.add_row(
                news['timestamp'],
                news['headline'],
                f"[{sentiment_color}]{news['sentiment']:.2f}[/{sentiment_color}]",
                f"{news['impact']:.2f}"
            )
        
        self.console.print(table)
    
    async def get_cpi_data(self):
        """Get and analyze CPI data"""
        self.console.print("[bold green]CPI DATA ANALYSIS[/bold green]")
        
        try:
            api_key = self.api_keys.get('FRED_API_KEY')
            if not api_key:
                self.console.print("[red]FRED API key not found![/red]")
                return
            
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'CPIAUCSL',
                'api_key': api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': 12
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            table = Table(title="CPI Data (Last 12 Months)")
            table.add_column("Date", style="cyan")
            table.add_column("CPI", style="green")
            table.add_column("YoY Change", style="yellow")
            
            observations = data['observations']
            for i in range(len(observations) - 1):
                current = float(observations[i]['value'])
                previous = float(observations[i + 1]['value'])
                yoy_change = ((current - previous) / previous) * 100
                
                table.add_row(
                    observations[i]['date'],
                    f"{current:.2f}",
                    f"{yoy_change:.2f}%"
                )
            
            self.console.print(table)
            
            # AI analysis
            self.console.print("\n[bold yellow]AI Analysis:[/bold yellow]")
            analysis = self.analyze_cpi_impact(observations)
            self.console.print(analysis)
            
        except Exception as e:
            self.console.print(f"[red]Error fetching CPI data: {str(e)}[/red]")
    
    async def signal_trading(self):
        """Generate AI trading signals"""
        self.console.print("[bold green]SIGNAL TRADING[/bold green]")
        
        # Top trading pairs
        coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT']
        selected_coins = np.random.choice(coins, 4, replace=False)
        
        table = Table(title="AI Trading Signals")
        table.add_column("Coin", style="cyan")
        table.add_column("Signal", style="green")
        table.add_column("Entry", style="yellow")
        table.add_column("TP", style="green")
        table.add_column("SL", style="red")
        table.add_column("Confidence", style="magenta")
        
        for coin in selected_coins:
            with self.console.status(f"Analyzing {coin}..."):
                market_data = await self.get_market_data(coin)
                news_sentiment = await self.analyze_news_sentiment(coin)
                on_chain_data = await self.get_on_chain_data(coin)
                economic_data = await self.get_economic_data()
                
                signal = self.ai_analyze_coin(coin, market_data, news_sentiment, on_chain_data, economic_data)
                
                signal_color = "green" if signal['direction'] == "LONG" else "red"
                table.add_row(
                    coin,
                    f"[{signal_color}]{signal['direction']}[/{signal_color}]",
                    f"${signal['entry_price']:.2f}",
                    f"${signal['take_profit']:.2f}",
                    f"${signal['stop_loss']:.2f}",
                    f"{signal['confidence']:.1f}%"
                )
                
                # Save signal to database
                self.save_signal(coin, signal)
        
        self.console.print(table)
    
    def save_signal(self, coin, signal):
        """Save trading signal to database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO signals (timestamp, coin, direction, entry_price, stop_loss, take_profit, confidence, reasoning, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            coin,
            signal['direction'],
            signal['entry_price'],
            signal['stop_loss'],
            signal['take_profit'],
            signal['confidence'],
            signal['reasoning'],
            'PENDING'
        ))
        
        conn.commit()
        conn.close()
    
    async def run(self):
        """Run the main application"""
        self.show_banner()
        
        while True:
            self.show_main_menu()
            choice = prompt("Select option (1-9): ")
            
            if choice == "1":
                await self.scan_coin()
            elif choice == "2":
                await self.news_scan()
            elif choice == "3":
                await self.get_cpi_data()
            elif choice == "4":
                await self.signal_trading()
            elif choice == "5":
                self.strategy_studio()
            elif choice == "6":
                self.backtest_engine()
            elif choice == "7":
                self.settings_menu()
            elif choice == "8":
                self.show_logs_stats()
            elif choice == "9":
                self.console.print("[bold green]Thank you for using Quentrade![/bold green]")
                break
            else:
                self.console.print("[red]Invalid option! Please try again.[/red]")
            
            time.sleep(1)
    
    # API Functions
    async def get_market_data(self, coin):
        """Get market data from Bybit"""
        # Implementation for Bybit API
        pass
    
    async def analyze_news_sentiment(self, coin):
        """Analyze news sentiment for specific coin"""
        # Implementation for news analysis
        pass
    
    async def get_on_chain_data(self, coin):
        """Get on-chain data from QuikNode"""
        # Implementation for on-chain data
        pass
    
    async def get_economic_data(self):
        """Get economic data from FRED"""
        # Implementation for economic data
        pass
    
    def ai_analyze_coin(self, coin, market_data, news_sentiment, on_chain_data, economic_data):
        """AI analysis combining all data sources"""
        # Implementation for AI analysis
        analysis = {
            'direction': 'LONG',
            'entry_price': 42000,
            'stop_loss': 40000,
            'take_profit': 45000,
            'confidence': 85.5,
            'risk_reward': 1.5,
            'reasoning': 'Bullish sentiment with strong on-chain metrics and positive macroeconomic outlook.'
        }
        return analysis
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def calculate_impact_score(self, headline, sentiment):
        """Calculate news impact score"""
        # Simple impact calculation
        return abs(sentiment) * 10
    
    def analyze_cpi_impact(self, cpi_data):
        """Analyze CPI impact on crypto market"""
        # Implementation for CPI analysis
        return "Current CPI trends suggest moderate inflation, which could be bullish for Bitcoin as a hedge against inflation."
    
    def strategy_studio(self):
        """Strategy creation and management"""
        self.console.print("[bold green]STRATEGY STUDIO[/bold green]")
        # Implementation for strategy studio
        pass
    
    def backtest_engine(self):
        """Backtest trading strategies"""
        self.console.print("[bold green]BACKTEST ENGINE[/bold green]")
        # Implementation for backtesting
        pass
    
    def settings_menu(self):
        """Settings and configuration"""
        self.console.print("[bold green]SETTINGS[/bold green]")
        # Implementation for settings
        pass
    
    def show_logs_stats(self):
        """Show logs and statistics"""
        self.console.print("[bold green]LOGS & STATISTICS[/bold green]")
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM signals ORDER BY timestamp DESC LIMIT 20")
        signals = cursor.fetchall()
        
        table = Table(title="Recent Signals")
        table.add_column("Time", style="cyan")
        table.add_column("Coin", style="green")
        table.add_column("Direction", style="yellow")
        table.add_column("Entry", style="white")
        table.add_column("Status", style="magenta")
        
        for signal in signals:
            table.add_row(
                signal[1],  # timestamp
                signal[2],  # coin
                signal[3],  # direction
                f"${signal[4]:.2f}",  # entry_price
                signal[8]   # status
            )
        
        self.console.print(table)
        conn.close()

if __name__ == "__main__":
    try:
        app = QuentradeAI()
        asyncio.run(app.run())
    except KeyboardInterrupt:
        console.print("\n[bold red]Application terminated by user.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logging.error(f"Application error: {str(e)}")