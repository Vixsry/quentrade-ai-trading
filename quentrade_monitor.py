#!/usr/bin/env python3
"""
Quentrade System Monitor
Monitors system health, API status, and trading performance
"""

import os
import sys
import psutil
import time
import asyncio
import requests
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
import sqlite3
from dotenv import load_dotenv

class QuentradeMonitor:
    def __init__(self):
        self.console = Console()
        load_dotenv()
        self.api_status = {}
        self.system_stats = {}
        self.trading_stats = {}
        self.db_file = "quentrade.db"
        
    def check_api_status(self):
        """Check status of all connected APIs"""
        apis = {
            'Bybit': 'https://api.bybit.com/v5/market/time',
            'CoinMarketCap': 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest',
            'FRED': 'https://api.stlouisfed.org/fred/series/observations',
            'NewsAPI': 'https://newsapi.org/v2/everything',
            'QuikNode': 'https://eth-mainnet.quiknode.pro/'
        }
        
        for name, url in apis.items():
            try:
                if name == 'CoinMarketCap':
                    headers = {'X-CMC_PRO_API_KEY': os.getenv('COINMARKETCAP_API_KEY')}
                    response = requests.get(url, headers=headers, timeout=5)
                elif name == 'FRED':
                    params = {
                        'series_id': 'CPIAUCSL',
                        'api_key': os.getenv('FRED_API_KEY'),
                        'file_type': 'json'
                    }
                    response = requests.get(url, params=params, timeout=5)
                elif name == 'NewsAPI':
                    params = {
                        'q': 'crypto',
                        'apiKey': os.getenv('NEWS_API_KEY')
                    }
                    response = requests.get(url, params=params, timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                
                self.api_status[name] = {
                    'status': '🟢 Online' if response.status_code == 200 else '🔴 Error',
                    'latency': f"{response.elapsed.total_seconds()*1000:.0f}ms",
                    'last_check': datetime.now().strftime('%H:%M:%S')
                }
            except Exception as e:
                self.api_status[name] = {
                    'status': '🔴 Offline',
                    'latency': 'N/A',
                    'last_check': datetime.now().strftime('%H:%M:%S')
                }
    
    def get_system_stats(self):
        """Get system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        self.system_stats = {
            'CPU Usage': f"{cpu_percent}%",
            'Memory Usage': f"{memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)",
            'Disk Usage': f"{disk.percent}% ({disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB)",
            'Python Memory': f"{psutil.Process().memory_info().rss / (1024**2):.1f} MB",
            'Uptime': self._get_uptime()
        }
    
    def get_trading_stats(self):
        """Get trading statistics from database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Get recent signals
            cursor.execute("""
                SELECT COUNT(*), 
                       SUM(CASE WHEN result > 0 THEN 1 ELSE 0 END),
                       AVG(result)
                FROM signals 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            total_signals, winning_signals, avg_result = cursor.fetchone()
            
            # Get active strategies
            cursor.execute("SELECT COUNT(*) FROM strategies WHERE active = 1")
            active_strategies = cursor.fetchone()[0]
            
            # Get recent news count
            cursor.execute("""
                SELECT COUNT(*) 
                FROM news_archive 
                WHERE timestamp > datetime('now', '-1 hour')
            """)
            recent_news = cursor.fetchone()[0]
            
            conn.close()
            
            self.trading_stats = {
                '24h Signals': total_signals or 0,
                'Win Rate': f"{(winning_signals/total_signals*100) if total_signals else 0:.1f}%",
                'Avg Result': f"{avg_result or 0:.2f}%",
                'Active Strategies': active_strategies,
                'News/Hour': recent_news
            }
            
        except Exception as e:
            self.trading_stats = {
                '24h Signals': 'Error',
                'Win Rate': 'Error',
                'Avg Result': 'Error',
                'Active Strategies': 'Error',
                'News/Hour': 'Error'
            }
    
    def _get_uptime(self):
        """Get system uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
                
            days = int(uptime_seconds // (24 * 3600))
            hours = int((uptime_seconds % (24 * 3600)) // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            
            return f"{days}d {hours}h {minutes}m"
        except:
            return "N/A"
    
    def create_dashboard(self):
        """Create monitoring dashboard"""
        layout = Layout()
        
        # Create main layout
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        # Split main area
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        # Header
        layout["header"].update(Panel(
            "[bold cyan]Quentrade System Monitor[/bold cyan]",
            border_style="cyan"
        ))
        
        # API Status Table
        api_table = Table(title="API Status", show_header=True, header_style="bold magenta")
        api_table.add_column("Service", style="cyan")
        api_table.add_column("Status", justify="center")
        api_table.add_column("Latency", justify="right")
        api_table.add_column("Last Check", justify="right")
        
        for name, status in self.api_status.items():
            api_table.add_row(
                name,
                status['status'],
                status['latency'],
                status['last_check']
            )
        
        layout["left"].update(Panel(api_table, border_style="green"))
        
        # System Stats Table
        system_table = Table(title="System Resources", show_header=True, header_style="bold magenta")
        system_table.add_column("Resource", style="cyan")
        system_table.add_column("Usage", justify="right")
        
        for resource, value in self.system_stats.items():
            system_table.add_row(resource, value)
        
        # Trading Stats Table
        trading_table = Table(title="Trading Statistics", show_header=True, header_style="bold magenta")
        trading_table.add_column("Metric", style="cyan")
        trading_table.add_column("Value", justify="right")
        
        for metric, value in self.trading_stats.items():
            trading_table.add_row(metric, str(value))
        
        # Combine system and trading stats
        combined_table = Table.grid(padding=1)
        combined_table.add_row(system_table)
        combined_table.add_row(trading_table)
        
        layout["right"].update(Panel(combined_table, border_style="blue"))
        
        # Footer
        layout["footer"].update(Panel(
            f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to exit",
            border_style="yellow"
        ))
        
        return layout
    
    async def run_monitor(self):
        """Run the monitoring dashboard"""
        with Live(self.create_dashboard(), refresh_per_second=1) as live:
            while True:
                self.check_api_status()
                self.get_system_stats()
                self.get_trading_stats()
                live.update(self.create_dashboard())
                await asyncio.sleep(5)  # Update every 5 seconds

def main():
    """Main function to run the monitor"""
    monitor = QuentradeMonitor()
    
    try:
        asyncio.run(monitor.run_monitor())
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()