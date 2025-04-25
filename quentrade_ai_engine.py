#!/usr/bin/env python3
"""
Quentrade AI Engine Module
Handles all AI-powered analysis and decision making
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

class QuentradeAIEngine:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.lstm_model = None
        self.sentiment_model = None
        self.risk_model = None
        self.strategy_model = None
        self.feature_scaler = None
        self.initialize_models()
        self.logger = logging.getLogger('QuentradeAI')
    
    def initialize_models(self):
        """Initialize or load AI models"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # Load or create LSTM model for price prediction
        self.lstm_model = self._load_or_create_lstm_model()
        
        # Load or create sentiment analysis model
        self.sentiment_model = self._load_or_create_sentiment_model()
        
        # Load or create risk assessment model
        self.risk_model = self._load_or_create_risk_model()
        
        # Load or create strategy selection model
        self.strategy_model = self._load_or_create_strategy_model()
        
        # Load feature scaler
        self.feature_scaler = self._load_or_create_scaler()
    
    def _load_or_create_lstm_model(self):
        """Load or create LSTM model for price prediction"""
        model_path = os.path.join(self.models_dir, "lstm_model.h5")
        
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        
        # Create new LSTM model
        model = keras.Sequential([
            keras.layers.LSTM(128, input_shape=(60, 20), return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(3, activation='softmax')  # [UP, DOWN, NEUTRAL]
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.save(model_path)
        return model
    
    def _load_or_create_sentiment_model(self):
        """Load or create sentiment analysis model"""
        model_path = os.path.join(self.models_dir, "sentiment_model.joblib")
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Create new sentiment model (simple for now)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        joblib.dump(model, model_path)
        return model
    
    def _load_or_create_risk_model(self):
        """Load or create risk assessment model"""
        model_path = os.path.join(self.models_dir, "risk_model.joblib")
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Create new risk model
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        joblib.dump(model, model_path)
        return model
    
    def _load_or_create_strategy_model(self):
        """Load or create strategy selection model"""
        model_path = os.path.join(self.models_dir, "strategy_model.joblib")
        
        if os.path.exists(model_path):
            return joblib.load(model_path)
        
        # Create new strategy model
        model = RandomForestClassifier(n_estimators=150, random_state=42)
        joblib.dump(model, model_path)
        return model
    
    def _load_or_create_scaler(self):
        """Load or create feature scaler"""
        scaler_path = os.path.join(self.models_dir, "feature_scaler.joblib")
        
        if os.path.exists(scaler_path):
            return joblib.load(scaler_path)
        
        # Create new scaler
        scaler = MinMaxScaler()
        joblib.dump(scaler, scaler_path)
        return scaler
    
    def analyze_coin(self, coin: str, market_data: Dict, news_data: List[Dict], 
                    on_chain_data: Dict, economic_data: Dict) -> Dict:
        """Comprehensive AI analysis of a coin"""
        
        # Prepare features
        features = self._prepare_features(market_data, news_data, on_chain_data, economic_data)
        
        # Get predictions from different models
        price_prediction = self._predict_price_movement(features)
        sentiment_score = self._analyze_sentiment(news_data)
        risk_assessment = self._assess_risk(features)
        optimal_strategy = self._select_strategy(features)
        
        # Generate trading signals
        signal = self._generate_signal(price_prediction, sentiment_score, risk_assessment)
        
        # Calculate entry, stop loss, and take profit
        prices = self._calculate_prices(market_data, signal, risk_assessment)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            price_prediction, sentiment_score, risk_assessment, 
            optimal_strategy, market_data, economic_data
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            price_prediction, sentiment_score, risk_assessment
        )
        
        return {
            "coin": coin,
            "direction": signal,
            "entry_price": prices['entry'],
            "stop_loss": prices['stop_loss'],
            "take_profit": prices['take_profit'],
            "confidence": confidence,
            "risk_level": risk_assessment['level'],
            "risk_reward": prices['risk_reward'],
            "reasoning": reasoning,
            "strategy": optimal_strategy,
            "timestamp": datetime.now().isoformat()
        }
    
    def _prepare_features(self, market_data: Dict, news_data: List[Dict], 
                         on_chain_data: Dict, economic_data: Dict) -> np.ndarray:
        """Prepare feature vector for AI models"""
        features = []
        
        # Market features
        if 'price' in market_data:
            features.extend([
                market_data['price'],
                market_data['volume_24h'],
                market_data['price_change_24h'],
                market_data.get('funding_rate', 0),
                market_data.get('orderbook_imbalance', 0.5)
            ])
        
        # News sentiment features
        news_sentiment = self._aggregate_news_sentiment(news_data)
        features.extend([
            news_sentiment['positive_ratio'],
            news_sentiment['negative_ratio'],
            news_sentiment['neutral_ratio'],
            news_sentiment['avg_polarity']
        ])
        
        # On-chain features (if available)
        if 'gas_price_gwei' in on_chain_data:
            features.extend([
                on_chain_data.get('gas_price_gwei', 0),
                on_chain_data.get('transactions_count', 0)
            ])
        
        # Economic features
        if 'CPI' in economic_data:
            features.extend([
                economic_data['CPI'].get('value', 0),
                economic_data['FED_RATE'].get('value', 0),
                economic_data['DXY'].get('value', 0)
            ])
        
        # Pad or truncate to fixed length
        target_length = 20
        if len(features) < target_length:
            features.extend([0] * (target_length - len(features)))
        elif len(features) > target_length:
            features = features[:target_length]
        
        return np.array(features).reshape(1, -1)
    
    def _predict_price_movement(self, features: np.ndarray) -> Dict:
        """Predict price movement using LSTM model"""
        # Reshape for LSTM input (samples, timesteps, features)
        lstm_input = features.reshape(1, 1, -1)
        
        # Repeat for 60 timesteps (simulate historical data)
        lstm_input = np.repeat(lstm_input, 60, axis=1)
        
        prediction = self.lstm_model.predict(lstm_input, verbose=0)[0]
        
        return {
            "up_prob": float(prediction[0]),
            "down_prob": float(prediction[1]),
            "neutral_prob": float(prediction[2])
        }
    
    def _analyze_sentiment(self, news_data: List[Dict]) -> Dict:
        """Analyze overall sentiment from news data"""
        sentiments = []
        
        for news in news_data:
            if 'sentiment' in news:
                sentiments.append(news['sentiment']['polarity'])
        
        if not sentiments:
            return {"score": 0, "classification": "NEUTRAL"}
        
        avg_sentiment = np.mean(sentiments)
        
        if avg_sentiment > 0.1:
            classification = "POSITIVE"
        elif avg_sentiment < -0.1:
            classification = "NEGATIVE"
        else:
            classification = "NEUTRAL"
        
        return {
            "score": float(avg_sentiment),
            "classification": classification
        }
    
    def _assess_risk(self, features: np.ndarray) -> Dict:
        """Assess risk level based on multiple factors"""
        # Simulate risk assessment (would use trained model in production)
        risk_score = np.random.uniform(0, 1)
        
        if risk_score < 0.3:
            level = "LOW"
        elif risk_score < 0.7:
            level = "MEDIUM"
        else:
            level = "HIGH"
        
        return {
            "score": float(risk_score),
            "level": level
        }
    
    def _select_strategy(self, features: np.ndarray) -> str:
        """Select optimal trading strategy"""
        strategies = [
            "TREND_FOLLOWING",
            "MEAN_REVERSION",
            "BREAKOUT",
            "SCALPING",
            "NEWS_BASED",
            "VOLATILITY_BREAKOUT"
        ]
        
        # Simulate strategy selection (would use trained model in production)
        strategy_idx = np.random.randint(0, len(strategies))
        return strategies[strategy_idx]
    
    def _generate_signal(self, price_prediction: Dict, sentiment: Dict, risk: Dict) -> str:
        """Generate trading signal based on all factors"""
        up_score = price_prediction['up_prob'] * 0.6
        
        if sentiment['classification'] == "POSITIVE":
            up_score += 0.2
        elif sentiment['classification'] == "NEGATIVE":
            up_score -= 0.2
        
        if risk['level'] == "HIGH":
            up_score -= 0.1
        
        if up_score > 0.55:
            return "LONG"
        elif up_score < 0.45:
            return "SHORT"
        else:
            return "NEUTRAL"
    
    def _calculate_prices(self, market_data: Dict, signal: str, risk_assessment: Dict) -> Dict:
        """Calculate entry, stop loss, and take profit prices"""
        current_price = market_data['price']
        
        # Adjust risk parameters based on risk assessment
        risk_multiplier = {
            "LOW": 1.2,
            "MEDIUM": 1.0,
            "HIGH": 0.8
        }[risk_assessment['level']]
        
        if signal == "LONG":
            stop_loss = current_price * (1 - 0.02 * risk_multiplier)
            take_profit = current_price * (1 + 0.04 * risk_multiplier)
        elif signal == "SHORT":
            stop_loss = current_price * (1 + 0.02 * risk_multiplier)
            take_profit = current_price * (1 - 0.04 * risk_multiplier)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            "entry": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_reward": risk_reward
        }
    
    def _generate_reasoning(self, price_prediction: Dict, sentiment: Dict, 
                          risk: Dict, strategy: str, market_data: Dict, 
                          economic_data: Dict) -> str:
        """Generate human-readable reasoning for the trading decision"""
        
        direction = "bullish" if price_prediction['up_prob'] > price_prediction['down_prob'] else "bearish"
        
        sentiment_desc = sentiment['classification'].lower()
        risk_desc = risk['level'].lower()
        
        reasoning = f"The AI analysis suggests a {direction} outlook with {price_prediction['up_prob']*100:.1f}% probability. "
        reasoning += f"Market sentiment is {sentiment_desc} based on recent news analysis. "
        reasoning += f"Risk level is assessed as {risk_desc}. "
        
        if 'price_change_24h' in market_data:
            reasoning += f"The asset has moved {market_data['price_change_24h']:.1f}% in the last 24 hours. "
        
        if 'CPI' in economic_data:
            reasoning += f"Current economic indicators show CPI at {economic_data['CPI']['value']:.2f}. "
        
        reasoning += f"Recommended strategy: {strategy.replace('_', ' ').title()}."
        
        return reasoning
    
    def _calculate_confidence(self, price_prediction: Dict, sentiment: Dict, 
                            risk: Dict) -> float:
        """Calculate overall confidence score"""
        price_confidence = max(price_prediction.values())
        sentiment_confidence = abs(sentiment['score'])
        risk_factor = 1 - (risk['score'] * 0.5)
        
        confidence = (price_confidence * 0.5 + sentiment_confidence * 0.3) * risk_factor
        return min(max(confidence * 100, 0), 100)
    
    def _aggregate_news_sentiment(self, news_data: List[Dict]) -> Dict:
        """Aggregate sentiment from multiple news items"""
        if not news_data:
            return {
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 1,
                "avg_polarity": 0
            }
        
        positive_count = sum(1 for news in news_data if news.get('sentiment', {}).get('classification') == 'POSITIVE')
        negative_count = sum(1 for news in news_data if news.get('sentiment', {}).get('classification') == 'NEGATIVE')
        neutral_count = len(news_data) - positive_count - negative_count
        
        polarities = [news.get('sentiment', {}).get('polarity', 0) for news in news_data]
        avg_polarity = np.mean(polarities) if polarities else 0
        
        total = len(news_data)
        
        return {
            "positive_ratio": positive_count / total,
            "negative_ratio": negative_count / total,
            "neutral_ratio": neutral_count / total,
            "avg_polarity": avg_polarity
        }
    
    def train_model(self, historical_data: pd.DataFrame, model_type: str = "lstm"):
        """Train AI model with historical data"""
        try:
            if model_type == "lstm":
                X, y = self._prepare_training_data(historical_data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.lstm_model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test, y_test),
                    verbose=1
                )
                
                # Save updated model
                model_path = os.path.join(self.models_dir, "lstm_model.h5")
                self.lstm_model.save(model_path)
                
                self.logger.info(f"LSTM model trained and saved to {model_path}")
                
            elif model_type == "sentiment":
                X, y = self._prepare_sentiment_data(historical_data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.sentiment_model.fit(X_train, y_train)
                
                # Save updated model
                model_path = os.path.join(self.models_dir, "sentiment_model.joblib")
                joblib.dump(self.sentiment_model, model_path)
                
                self.logger.info(f"Sentiment model trained and saved to {model_path}")
                
            elif model_type == "risk":
                X, y = self._prepare_risk_data(historical_data)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.risk_model.fit(X_train, y_train)
                
                # Save updated model
                model_path = os.path.join(self.models_dir, "risk_model.joblib")
                joblib.dump(self.risk_model, model_path)
                
                self.logger.info(f"Risk model trained and saved to {model_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model: {str(e)}")
            return False
    
    def _prepare_training_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        # Feature engineering
        features = []
        targets = []
        
        # Assuming historical_data has columns like: price, volume, sentiment, etc.
        required_columns = ['price', 'volume', 'price_change', 'sentiment_score']
        
        for col in required_columns:
            if col not in historical_data.columns:
                historical_data[col] = 0
        
        # Create sequences
        sequence_length = 60
        
        for i in range(len(historical_data) - sequence_length - 1):
            sequence = historical_data.iloc[i:i+sequence_length][required_columns].values
            target = historical_data.iloc[i+sequence_length]['price_change']
            
            features.append(sequence)
            
            # Convert target to one-hot encoding [UP, DOWN, NEUTRAL]
            if target > 0.01:
                targets.append([1, 0, 0])  # UP
            elif target < -0.01:
                targets.append([0, 1, 0])  # DOWN
            else:
                targets.append([0, 0, 1])  # NEUTRAL
        
        return np.array(features), np.array(targets)
    
    def _prepare_sentiment_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sentiment model training"""
        # Implement sentiment data preparation
        return np.array([]), np.array([])
    
    def _prepare_risk_data(self, historical_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for risk model training"""
        # Implement risk data preparation
        return np.array([]), np.array([])
    
    def backtest_strategy(self, strategy_name: str, historical_data: pd.DataFrame) -> Dict:
        """Backtest a trading strategy on historical data"""
        results = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "trades": []
        }
        
        capital = 10000  # Starting capital
        position = None
        highest_capital = capital
        
        for i in range(60, len(historical_data)):
            # Get features for analysis
            features = self._extract_features_for_backtest(historical_data, i)
            
            # Generate signal
            signal = self._generate_backtest_signal(features, strategy_name)
            
            current_price = historical_data.iloc[i]['price']
            
            # Close position if conditions met
            if position is not None:
                if (position['type'] == 'LONG' and signal == 'SHORT') or \
                   (position['type'] == 'SHORT' and signal == 'LONG') or \
                   (current_price <= position['stop_loss']) or \
                   (current_price >= position['take_profit']):
                    
                    # Calculate profit/loss
                    if position['type'] == 'LONG':
                        profit = (current_price - position['entry_price']) * position['quantity']
                    else:
                        profit = (position['entry_price'] - current_price) * position['quantity']
                    
                    capital += profit
                    results['total_trades'] += 1
                    
                    if profit > 0:
                        results['winning_trades'] += 1
                    else:
                        results['losing_trades'] += 1
                    
                    results['total_profit'] += profit
                    
                    # Record trade
                    results['trades'].append({
                        'entry_time': position['entry_time'],
                        'exit_time': historical_data.iloc[i]['timestamp'],
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'profit': profit
                    })
                    
                    position = None
                    
                    # Update highest capital for drawdown calculation
                    if capital > highest_capital:
                        highest_capital = capital
            
            # Open new position if no position exists
            if position is None and signal in ['LONG', 'SHORT']:
                risk_per_trade = capital * 0.02  # Risk 2% per trade
                stop_loss_distance = current_price * 0.01  # 1% stop loss
                quantity = risk_per_trade / stop_loss_distance
                
                position = {
                    'type': signal,
                    'entry_price': current_price,
                    'entry_time': historical_data.iloc[i]['timestamp'],
                    'quantity': quantity,
                    'stop_loss': current_price * (0.99 if signal == 'LONG' else 1.01),
                    'take_profit': current_price * (1.02 if signal == 'LONG' else 0.98)
                }
            
            # Calculate drawdown
            drawdown = (highest_capital - capital) / highest_capital
            if drawdown > results['max_drawdown']:
                results['max_drawdown'] = drawdown
        
        # Calculate final metrics
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades']
            
            gross_profit = sum(trade['profit'] for trade in results['trades'] if trade['profit'] > 0)
            gross_loss = abs(sum(trade['profit'] for trade in results['trades'] if trade['profit'] < 0))
            
            if gross_loss > 0:
                results['profit_factor'] = gross_profit / gross_loss
            else:
                results['profit_factor'] = float('inf')
        
        return results
    
    def _extract_features_for_backtest(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Extract features for backtesting at specific index"""
        # Get the last 60 data points
        window_data = data.iloc[max(0, index-60):index]
        
        features = []
        
        # Price and volume features
        features.append(data.iloc[index]['price'])
        features.append(data.iloc[index]['volume'])
        
        # Technical indicators
        if 'sma_20' in data.columns:
            features.append(data.iloc[index]['sma_20'])
        if 'rsi_14' in data.columns:
            features.append(data.iloc[index]['rsi_14'])
        
        # Add more features as needed
        
        return np.array(features).reshape(1, -1)
    
    def _generate_backtest_signal(self, features: np.ndarray, strategy_name: str) -> str:
        """Generate trading signal for backtesting"""
        # Use AI model to generate signal based on strategy
        if strategy_name == "TREND_FOLLOWING":
            # Implement trend following logic
            return "LONG"  # Placeholder
        elif strategy_name == "MEAN_REVERSION":
            # Implement mean reversion logic
            return "SHORT"  # Placeholder
        # Add more strategies
        
        return "NEUTRAL"
    
    def create_strategy(self, name: str, description: str, rules: Dict) -> bool:
        """Create a new trading strategy"""
        try:
            # Validate strategy rules
            required_rules = ['entry_conditions', 'exit_conditions', 'risk_management']
            for rule in required_rules:
                if rule not in rules:
                    raise ValueError(f"Missing required rule: {rule}")
            
            # Save strategy to database/file
            strategy_path = os.path.join(self.models_dir, f"strategy_{name.lower()}.json")
            
            strategy_data = {
                'name': name,
                'description': description,
                'rules': rules,
                'created_at': datetime.now().isoformat(),
                'active': True
            }
            
            with open(strategy_path, 'w') as f:
                json.dump(strategy_data, f, indent=4)
            
            self.logger.info(f"Strategy '{name}' created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating strategy: {str(e)}")
            return False
    
    def optimize_parameters(self, strategy_name: str, historical_data: pd.DataFrame, 
                          param_ranges: Dict) -> Dict:
        """Optimize strategy parameters using grid search"""
        best_params = None
        best_performance = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        
        for params in param_combinations:
            # Apply parameters to strategy
            self._apply_strategy_params(strategy_name, params)
            
            # Backtest with current parameters
            results = self.backtest_strategy(strategy_name, historical_data)
            
            # Use profit factor as optimization metric
            performance = results['profit_factor']
            
            if performance > best_performance:
                best_performance = performance
                best_params = params.copy()
        
        return {
            'best_params': best_params,
            'best_performance': best_performance
        }
    
    def _generate_param_combinations(self, param_ranges: Dict) -> List[Dict]:
        """Generate all possible parameter combinations"""
        import itertools
        
        keys = param_ranges.keys()
        values = param_ranges.values()
        
        combinations = []
        for values_combo in itertools.product(*values):
            combinations.append(dict(zip(keys, values_combo)))
        
        return combinations
    
    def _apply_strategy_params(self, strategy_name: str, params: Dict):
        """Apply parameters to a strategy"""
        # Implement parameter application logic
        pass
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted report from analysis results"""
        report = f"""
Quentrade AI Analysis Report
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Asset: {results['coin']}
Signal: {results['direction']}
Confidence: {results['confidence']:.1f}%

Price Levels:
- Entry: ${results['entry_price']:.2f}
- Stop Loss: ${results['stop_loss']:.2f}
- Take Profit: ${results['take_profit']:.2f}
- Risk/Reward: 1:{results['risk_reward']:.1f}

Risk Assessment: {results['risk_level']}
Recommended Strategy: {results['strategy']}

AI Reasoning:
{results['reasoning']}

Note: This is an AI-generated analysis. Always use proper risk management and do your own research.
"""
        return report
    
    def save_model_state(self):
        """Save current state of all models"""
        try:
            # Save LSTM model
            self.lstm_model.save(os.path.join(self.models_dir, "lstm_model.h5"))
            
            # Save other models
            joblib.dump(self.sentiment_model, os.path.join(self.models_dir, "sentiment_model.joblib"))
            joblib.dump(self.risk_model, os.path.join(self.models_dir, "risk_model.joblib"))
            joblib.dump(self.strategy_model, os.path.join(self.models_dir, "strategy_model.joblib"))
            joblib.dump(self.feature_scaler, os.path.join(self.models_dir, "feature_scaler.joblib"))
            
            self.logger.info("All models saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            return False