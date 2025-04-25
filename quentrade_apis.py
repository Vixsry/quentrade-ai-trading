#!/usr/bin/env python3
"""
Quentrade API Integration Module
Handles all external API connections and data fetching
"""

import os
import time
import requests
import hmac
import hashlib
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Optional
from pybit.unified_trading import HTTP
import ccxt
from fredapi import Fred
import yfinance as yf
from textblob import TextBlob
import tweepy
import feedparser
import aiohttp
import asyncio

class QuentradeAPIs:
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.bybit_client = None
        self.ccxt_client = None
        self.fred_client = None
        self.news_sources = self._initialize_news_sources()
        self.initialize_clients()
    
    def initialize_clients(self):
        """Initialize all API clients"""
        # Bybit client
        if self.api_keys.get('BYBIT_API_KEY') and self.api_keys.get('BYBIT_API_SECRET'):
            try:
                self.bybit_client = HTTP(
                    testnet=False,
                    api_key=self.api_keys['BYBIT_API_KEY'],
                    api_secret=self.api_keys['BYBIT_API_SECRET']
                )
                self.ccxt_client = ccxt.bybit({
                    'apiKey': self.api_keys['BYBIT_API_KEY'],
                    'secret': self.api_keys['BYBIT_API_SECRET']
                })
            except Exception as e:
            return {"error": f"Failed to fetch market data: {str(e)}"}
    
    def _calculate_orderbook_imbalance(self, orderbook_data: Dict) -> float:
        """Calculate orderbook imbalance ratio"""
        try:
            bids_volume = sum([float(bid[1]) for bid in orderbook_data['b'][:10]])
            asks_volume = sum([float(ask[1]) for ask in orderbook_data['a'][:10]])
            
            if asks_volume == 0:
                return 1.0
            
            return bids_volume / asks_volume
        
        except Exception:
            return 0.0
    
    async def get_on_chain_data(self, symbol: str) -> Dict:
        """Get on-chain data from QuikNode"""
        try:
            if not self.api_keys.get('QUIKNODE_API_KEY'):
                return {"error": "QuikNode API key not configured"}
            
            # Extract base currency from symbol (e.g., BTC from BTCUSDT)
            base = symbol[:-4] if symbol.endswith("USDT") else symbol
            
            # QuikNode endpoint for Ethereum data
            if base in ["ETH", "USDC", "USDT", "DAI", "LINK", "UNI"]:
                headers = {
                    "Content-Type": "application/json",
                    "x-qn-api-key": self.api_keys['QUIKNODE_API_KEY']
                }
                
                # Get various on-chain metrics
                metrics = {}
                
                # Gas fees
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_gasPrice",
                    "params": []
                }
                
                response = requests.post(
                    "https://eth-mainnet.quiknode.pro/",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    gas_price = int(response.json()['result'], 16) / 1e9  # Convert to Gwei
                    metrics['gas_price_gwei'] = gas_price
                
                # Get latest block
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_getBlockByNumber",
                    "params": ["latest", True]
                }
                
                response = requests.post(
                    "https://eth-mainnet.quiknode.pro/",
                    headers=headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    block_data = response.json()['result']
                    metrics['block_number'] = int(block_data['number'], 16)
                    metrics['transactions_count'] = len(block_data['transactions'])
                
                return metrics
            
            # For Bitcoin
            elif base == "BTC":
                # Bitcoin-specific metrics would go here
                return {"info": "Bitcoin on-chain data not implemented yet"}
            
            else:
                return {"info": f"No on-chain data available for {base}"}
        
        except Exception as e:
            return {"error": f"Failed to fetch on-chain data: {str(e)}"}
    
    async def get_coinmarketcap_data(self, symbol: str) -> Dict:
        """Get fundamental data from CoinMarketCap"""
        try:
            if not self.api_keys.get('COINMARKETCAP_API_KEY'):
                return {"error": "CoinMarketCap API key not configured"}
            
            base = symbol[:-4] if symbol.endswith("USDT") else symbol
            
            headers = {
                'Accepts': 'application/json',
                'X-CMC_PRO_API_KEY': self.api_keys['COINMARKETCAP_API_KEY'],
            }
            
            # Get latest quotes
            url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
            parameters = {
                'symbol': base,
                'convert': 'USD'
            }
            
            response = requests.get(url, headers=headers, params=parameters)
            data = response.json()
            
            if 'data' in data and base in data['data']:
                coin_data = data['data'][base]
                quote = coin_data['quote']['USD']
                
                return {
                    "name": coin_data['name'],
                    "rank": coin_data['cmc_rank'],
                    "market_cap": quote['market_cap'],
                    "volume_24h": quote['volume_24h'],
                    "circulating_supply": coin_data['circulating_supply'],
                    "total_supply": coin_data['total_supply'],
                    "max_supply": coin_data['max_supply'],
                    "percent_change_1h": quote['percent_change_1h'],
                    "percent_change_24h": quote['percent_change_24h'],
                    "percent_change_7d": quote['percent_change_7d'],
                    "percent_change_30d": quote['percent_change_30d'],
                    "market_dominance": quote.get('market_cap_dominance', 0)
                }
            else:
                return {"error": f"No data found for {base}"}
        
        except Exception as e:
            return {"error": f"Failed to fetch CoinMarketCap data: {str(e)}"}
    
    async def get_economic_data(self) -> Dict:
        """Get economic indicators from FRED"""
        try:
            if not self.fred_client:
                return {"error": "FRED client not initialized"}
            
            indicators = {
                'CPI': 'CPIAUCSL',           # Consumer Price Index
                'FED_RATE': 'FEDFUNDS',      # Federal Funds Rate
                'UNEMPLOYMENT': 'UNRATE',     # Unemployment Rate
                'GDP': 'GDP',                # Gross Domestic Product
                'DXY': 'DTWEXBGS',           # US Dollar Index
                'M2': 'M2SL',                # Money Supply M2
                'VIX': 'VIXCLS',             # CBOE Volatility Index
                'TREASURY_10Y': 'DGS10',     # 10-Year Treasury Rate
                'INFLATION_EXPECT': 'T5YIE',  # 5-Year Breakeven Inflation Rate
                'PCE': 'PCEPI'               # Personal Consumption Expenditures
            }
            
            economic_data = {}
            
            for name, series_id in indicators.items():
                try:
                    data = self.fred_client.get_series_latest_release(series_id)
                    if not data.empty:
                        latest_value = data.iloc[-1]
                        previous_value = data.iloc[-2] if len(data) > 1 else None
                        
                        change = ((latest_value - previous_value) / previous_value * 100) if previous_value else None
                        
                        economic_data[name] = {
                            'value': float(latest_value),
                            'date': data.index[-1].strftime('%Y-%m-%d'),
                            'change': float(change) if change else None
                        }
                except Exception:
                    continue
            
            return economic_data
        
        except Exception as e:
            return {"error": f"Failed to fetch economic data: {str(e)}"}
    
    async def fetch_news(self, coin: Optional[str] = None) -> List[Dict]:
        """Fetch and analyze news from multiple sources"""
        news_items = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.news_sources:
                tasks.append(self._fetch_single_source(session, source, coin))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    news_items.extend(result)
        
        # Sort by timestamp and return top 50
        news_items.sort(key=lambda x: x['timestamp'], reverse=True)
        return news_items[:50]
    
    async def _fetch_single_source(self, session: aiohttp.ClientSession, source: Dict, coin: Optional[str]) -> List[Dict]:
        """Fetch news from a single source"""
        try:
            if source['type'] == 'rss':
                async with session.get(source['url']) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        items = []
                        for entry in feed.entries[:10]:  # Get top 10 from each source
                            # Filter by coin if specified
                            if coin and coin.lower() not in entry.title.lower():
                                continue
                            
                            # Analyze sentiment
                            sentiment = self._analyze_sentiment(entry.title + ' ' + entry.get('summary', ''))
                            
                            items.append({
                                'source': source['name'],
                                'title': entry.title,
                                'link': entry.link,
                                'timestamp': datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now(),
                                'sentiment': sentiment,
                                'summary': entry.get('summary', '')[:200]
                            })
                        
                        return items
            
            return []
        
        except Exception:
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment
            
            # Classify sentiment
            if sentiment.polarity > 0.1:
                classification = 'POSITIVE'
            elif sentiment.polarity < -0.1:
                classification = 'NEGATIVE'
            else:
                classification = 'NEUTRAL'
            
            return {
                'polarity': sentiment.polarity,
                'subjectivity': sentiment.subjectivity,
                'classification': classification
            }
        
        except Exception:
            return {
                'polarity': 0.0,
                'subjectivity': 0.0,
                'classification': 'NEUTRAL'
            }
    
    async def get_fear_greed_index(self) -> Dict:
        """Get crypto fear and greed index"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://api.alternative.me/fng/') as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            latest = data['data'][0]
                            return {
                                'value': int(latest['value']),
                                'classification': latest['value_classification'],
                                'timestamp': datetime.fromtimestamp(int(latest['timestamp']))
                            }
            
            return {"error": "Failed to fetch fear and greed index"}
        
        except Exception as e:
            return {"error": f"Error fetching fear and greed index: {str(e)}"}
    
    async def get_social_sentiment(self, coin: str) -> Dict:
        """Get social sentiment from various platforms (simplified version)"""
        # This would ideally connect to Twitter API, Reddit API, etc.
        # For now, returning simulated data
        return {
            "twitter_mentions": np.random.randint(1000, 10000),
            "reddit_mentions": np.random.randint(100, 1000),
            "social_sentiment": np.random.uniform(-1, 1),
            "trending_score": np.random.uniform(0, 100)
        }
    
    def execute_trade(self, symbol: str, side: str, quantity: float, price: Optional[float] = None, 
                     stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> Dict:
        """Execute a trade on Bybit"""
        try:
            if not self.bybit_client:
                return {"error": "Bybit client not initialized"}
            
            # Create order
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": side.capitalize(),
                "orderType": "Market" if price is None else "Limit",
                "qty": str(quantity),
                "timeInForce": "GTC",
                "reduceOnly": False,
                "closeOnTrigger": False
            }
            
            if price:
                order_params["price"] = str(price)
            
            result = self.bybit_client.place_order(**order_params)
            
            if result['retCode'] == 0:
                order_id = result['result']['orderId']
                
                # Set stop loss and take profit if provided
                if stop_loss or take_profit:
                    sl_tp_params = {
                        "category": "linear",
                        "symbol": symbol,
                        "positionIdx": 0
                    }
                    
                    if stop_loss:
                        sl_tp_params["stopLoss"] = str(stop_loss)
                    
                    if take_profit:
                        sl_tp_params["takeProfit"] = str(take_profit)
                    
                    self.bybit_client.set_trading_stop(**sl_tp_params)
                
                return {
                    "status": "success",
                    "order_id": order_id,
                    "message": f"{side} order placed successfully"
                }
            else:
                return {
                    "status": "error",
                    "message": result['retMsg']
                }
        
        except Exception as e:
            return {"error": f"Failed to execute trade: {str(e)}"}:
                print(f"Error initializing Bybit client: {e}")
        
        # FRED client for economic data
        if self.api_keys.get('FRED_API_KEY'):
            try:
                self.fred_client = Fred(api_key=self.api_keys['FRED_API_KEY'])
            except Exception as e:
                print(f"Error initializing FRED client: {e}")
    
    def _initialize_news_sources(self) -> List[Dict]:
        """Initialize news sources for scraping"""
        return [
            {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "type": "rss"},
            {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss", "type": "rss"},
            {"name": "CryptoNews", "url": "https://cryptonews.com/news/feed/", "type": "rss"},
            {"name": "Bitcoin Magazine", "url": "https://bitcoinmagazine.com/.rss/full/", "type": "rss"},
            {"name": "The Block", "url": "https://www.theblockcrypto.com/rss.xml", "type": "rss"},
            {"name": "Decrypt", "url": "https://decrypt.co/feed", "type": "rss"},
            {"name": "CryptoPotato", "url": "https://cryptopotato.com/feed/", "type": "rss"},
            {"name": "Bitcoinist", "url": "https://bitcoinist.com/feed/", "type": "rss"},
            {"name": "U.Today", "url": "https://u.today/rss", "type": "rss"},
            {"name": "CryptoBriefing", "url": "https://cryptobriefing.com/feed/", "type": "rss"},
            {"name": "AMBCrypto", "url": "https://ambcrypto.com/feed/", "type": "rss"},
            {"name": "NewsBTC", "url": "https://www.newsbtc.com/feed/", "type": "rss"},
            {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/", "type": "rss"},
            {"name": "Crypto Daily", "url": "https://cryptodaily.co.uk/feed", "type": "rss"},
            {"name": "BeInCrypto", "url": "https://beincrypto.com/feed/", "type": "rss"}
        ]
    
    async def get_market_data(self, symbol: str) -> Dict:
        """Get comprehensive market data for a symbol"""
        try:
            if not self.bybit_client:
                return {"error": "Bybit client not initialized"}
            
            # Get ticker data
            ticker = self.bybit_client.get_tickers(
                category="linear",
                symbol=symbol
            )
            
            # Get kline data (1h candles for the last 24 hours)
            klines = self.bybit_client.get_kline(
                category="linear",
                symbol=symbol,
                interval="60",
                limit=24
            )
            
            # Get orderbook
            orderbook = self.bybit_client.get_orderbook(
                category="linear",
                symbol=symbol,
                limit=50
            )
            
            # Get recent trades
            trades = self.bybit_client.get_public_trade_history(
                category="linear",
                symbol=symbol,
                limit=100
            )
            
            # Get funding rate
            funding = self.bybit_client.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=1
            )
            
            # Process and combine data
            result = {
                "symbol": symbol,
                "price": float(ticker['result']['list'][0]['lastPrice']),
                "volume_24h": float(ticker['result']['list'][0]['volume24h']),
                "price_change_24h": float(ticker['result']['list'][0]['price24hPcnt']) * 100,
                "high_24h": float(ticker['result']['list'][0]['highPrice24h']),
                "low_24h": float(ticker['result']['list'][0]['lowPrice24h']),
                "funding_rate": float(funding['result']['list'][0]['fundingRate']) * 100 if funding['result']['list'] else 0,
                "bid_ask_spread": float(orderbook['result']['b'][0][0]) - float(orderbook['result']['a'][0][0]),
                "klines": klines['result']['list'],
                "orderbook_imbalance": self._calculate_orderbook_imbalance(orderbook['result'])
            }
            
            return result
            
        except Exception as e