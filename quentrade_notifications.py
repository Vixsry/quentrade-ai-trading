#!/usr/bin/env python3
"""
Quentrade Notifications Module
Handles alerts and notifications through various channels
"""

import os
import json
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import telegram
import discord
from discord.webhook import SyncWebhook
import logging
import asyncio
from typing import Dict, Optional
from dotenv import load_dotenv

class QuentradeNotifications:
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger('QuentradeNotifications')
        
        # Telegram settings
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_bot = None
        if self.telegram_token and self.telegram_chat_id:
            try:
                self.telegram_bot = telegram.Bot(token=self.telegram_token)
            except Exception as e:
                self.logger.error(f"Failed to initialize Telegram bot: {e}")
        
        # Discord settings
        self.discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
        
        # Email settings
        self.email_enabled = os.getenv('EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
        self.smtp_server = os.getenv('SMTP_SERVER')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME')
        self.smtp_password = os.getenv('SMTP_PASSWORD')
        self.email_from = os.getenv('EMAIL_FROM')
        self.email_to = os.getenv('EMAIL_TO')
    
    async def send_telegram_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message via Telegram"""
        if not self.telegram_bot:
            self.logger.warning("Telegram bot not configured")
            return False
        
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_discord_message(self, message: str, embed: Optional[Dict] = None) -> bool:
        """Send message via Discord webhook"""
        if not self.discord_webhook_url:
            self.logger.warning("Discord webhook not configured")
            return False
        
        try:
            webhook = SyncWebhook.from_url(self.discord_webhook_url)
            
            if embed:
                discord_embed = discord.Embed(
                    title=embed.get('title', ''),
                    description=embed.get('description', ''),
                    color=embed.get('color', 0x00ff00)
                )
                
                for field in embed.get('fields', []):
                    discord_embed.add_field(
                        name=field['name'],
                        value=field['value'],
                        inline=field.get('inline', False)
                    )
                
                webhook.send(embed=discord_embed)
            else:
                webhook.send(content=message)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send Discord message: {e}")
            return False
    
    def send_email(self, subject: str, body: str, html: bool = False) -> bool:
        """Send email notification"""
        if not self.email_enabled:
            self.logger.warning("Email notifications not enabled")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_from
            msg['To'] = self.email_to
            msg['Subject'] = subject
            
            if html:
                msg.attach(MIMEText(body, 'html'))
            else:
                msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")
            return False
    
    def format_signal_alert(self, signal: Dict) -> Dict:
        """Format trading signal for notifications"""
        emoji_map = {
            'LONG': '🟢',
            'SHORT': '🔴',
            'NEUTRAL': '⚪'
        }
        
        telegram_message = f"""
<b>{emoji_map.get(signal['direction'], '⚪')} NEW TRADING SIGNAL</b>

<b>Coin:</b> {signal['coin']}
<b>Direction:</b> {signal['direction']}
<b>Entry Price:</b> ${signal['entry_price']:.2f}
<b>Stop Loss:</b> ${signal['stop_loss']:.2f}
<b>Take Profit:</b> ${signal['take_profit']:.2f}
<b>Confidence:</b> {signal['confidence']:.1f}%
<b>Risk/Reward:</b> 1:{signal['risk_reward']:.1f}

<b>Reasoning:</b> {signal['reasoning']}

<i>Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
"""
        
        discord_embed = {
            'title': f"{emoji_map.get(signal['direction'], '⚪')} Trading Signal - {signal['coin']}",
            'description': signal['reasoning'],
            'color': 0x00ff00 if signal['direction'] == 'LONG' else 0xff0000 if signal['direction'] == 'SHORT' else 0xffffff,
            'fields': [
                {'name': 'Direction', 'value': signal['direction'], 'inline': True},
                {'name': 'Entry Price', 'value': f"${signal['entry_price']:.2f}", 'inline': True},
                {'name': 'Stop Loss', 'value': f"${signal['stop_loss']:.2f}", 'inline': True},
                {'name': 'Take Profit', 'value': f"${signal['take_profit']:.2f}", 'inline': True},
                {'name': 'Confidence', 'value': f"{signal['confidence']:.1f}%", 'inline': True},
                {'name': 'Risk/Reward', 'value': f"1:{signal['risk_reward']:.1f}", 'inline': True}
            ]
        }
        
        email_subject = f"Quentrade Signal: {signal['direction']} {signal['coin']}"
        
        email_body = f"""
        <h2>New Trading Signal</h2>
        <p><strong>Coin:</strong> {signal['coin']}</p>
        <p><strong>Direction:</strong> {signal['direction']}</p>
        <p><strong>Entry Price:</strong> ${signal['entry_price']:.2f}</p>
        <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.2f}</p>
        <p><strong>Take Profit:</strong> ${signal['take_profit']:.2f}</p>
        <p><strong>Confidence:</strong> {signal['confidence']:.1f}%</p>
        <p><strong>Risk/Reward:</strong> 1:{signal['risk_reward']:.1f}</p>
        <br>
        <p><strong>Reasoning:</strong> {signal['reasoning']}</p>
        <br>
        <p><em>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        """
        
        return {
            'telegram': telegram_message,
            'discord': discord_embed,
            'email_subject': email_subject,
            'email_body': email_body
        }
    
    def format_error_alert(self, error_type: str, error_message: str, details: Dict = None) -> Dict:
        """Format error alerts for notifications"""
        telegram_message = f"""
<b>⚠️ QUENTRADE ERROR ALERT</b>

<b>Type:</b> {error_type}
<b>Message:</b> {error_message}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        if details:
            telegram_message += "\n<b>Details:</b>\n"
            for key, value in details.items():
                telegram_message += f"- {key}: {value}\n"
        
        discord_embed = {
            'title': f"⚠️ Error Alert - {error_type}",
            'description': error_message,
            'color': 0xff0000,
            'fields': []
        }
        
        if details:
            for key, value in details.items():
                discord_embed['fields'].append({
                    'name': key,
                    'value': str(value),
                    'inline': True
                })
        
        email_subject = f"Quentrade Error: {error_type}"
        
        email_body = f"""
        <h2>Error Alert</h2>
        <p><strong>Type:</strong> {error_type}</p>
        <p><strong>Message:</strong> {error_message}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        if details:
            email_body += "<h3>Details:</h3><ul>"
            for key, value in details.items():
                email_body += f"<li><strong>{key}:</strong> {value}</li>"
            email_body += "</ul>"
        
        return {
            'telegram': telegram_message,
            'discord': discord_embed,
            'email_subject': email_subject,
            'email_body': email_body
        }
    
    def format_status_update(self, status_type: str, metrics: Dict) -> Dict:
        """Format status updates for notifications"""
        telegram_message = f"""
<b>📊 QUENTRADE STATUS UPDATE</b>

<b>Type:</b> {status_type}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Metrics:</b>
"""
        
        for key, value in metrics.items():
            telegram_message += f"• {key}: {value}\n"
        
        discord_embed = {
            'title': f"📊 Status Update - {status_type}",
            'color': 0x3498db,
            'fields': []
        }
        
        for key, value in metrics.items():
            discord_embed['fields'].append({
                'name': key,
                'value': str(value),
                'inline': True
            })
        
        email_subject = f"Quentrade Status: {status_type}"
        
        email_body = f"""
        <h2>Status Update</h2>
        <p><strong>Type:</strong> {status_type}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h3>Metrics:</h3>
        <ul>
        """
        
        for key, value in metrics.items():
            email_body += f"<li><strong>{key}:</strong> {value}</li>"
        email_body += "</ul>"
        
        return {
            'telegram': telegram_message,
            'discord': discord_embed,
            'email_subject': email_subject,
            'email_body': email_body
        }
    
    async def send_signal_alert(self, signal: Dict) -> Dict:
        """Send trading signal alerts to all configured channels"""
        formatted = self.format_signal_alert(signal)
        results = {
            'telegram': False,
            'discord': False,
            'email': False
        }
        
        # Send Telegram
        if self.telegram_bot:
            results['telegram'] = await self.send_telegram_message(formatted['telegram'])
        
        # Send Discord
        if self.discord_webhook_url:
            results['discord'] = self.send_discord_message("", embed=formatted['discord'])
        
        # Send Email
        if self.email_enabled:
            results['email'] = self.send_email(
                formatted['email_subject'],
                formatted['email_body'],
                html=True
            )
        
        return results
    
    async def send_error_alert(self, error_type: str, error_message: str, details: Dict = None) -> Dict:
        """Send error alerts to all configured channels"""
        formatted = self.format_error_alert(error_type, error_message, details)
        results = {
            'telegram': False,
            'discord': False,
            'email': False
        }
        
        # Send Telegram
        if self.telegram_bot:
            results['telegram'] = await self.send_telegram_message(formatted['telegram'])
        
        # Send Discord
        if self.discord_webhook_url:
            results['discord'] = self.send_discord_message("", embed=formatted['discord'])
        
        # Send Email
        if self.email_enabled:
            results['email'] = self.send_email(
                formatted['email_subject'],
                formatted['email_body'],
                html=True
            )
        
        return results
    
    async def send_status_update(self, status_type: str, metrics: Dict) -> Dict:
        """Send status updates to all configured channels"""
        formatted = self.format_status_update(status_type, metrics)
        results = {
            'telegram': False,
            'discord': False,
            'email': False
        }
        
        # Send Telegram
        if self.telegram_bot:
            results['telegram'] = await self.send_telegram_message(formatted['telegram'])
        
        # Send Discord
        if self.discord_webhook_url:
            results['discord'] = self.send_discord_message("", embed=formatted['discord'])
        
        # Send Email
        if self.email_enabled:
            results['email'] = self.send_email(
                formatted['email_subject'],
                formatted['email_body'],
                html=True
            )
        
        return results
    
    def format_trade_execution(self, trade: Dict) -> Dict:
        """Format trade execution notification"""
        telegram_message = f"""
<b>✅ TRADE EXECUTED</b>

<b>Symbol:</b> {trade['symbol']}
<b>Side:</b> {trade['side']}
<b>Price:</b> ${trade['price']:.2f}
<b>Quantity:</b> {trade['quantity']}
<b>Order ID:</b> {trade['order_id']}
<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        discord_embed = {
            'title': f"✅ Trade Executed - {trade['symbol']}",
            'color': 0x00ff00 if trade['side'].upper() == 'BUY' else 0xff0000,
            'fields': [
                {'name': 'Side', 'value': trade['side'], 'inline': True},
                {'name': 'Price', 'value': f"${trade['price']:.2f}", 'inline': True},
                {'name': 'Quantity', 'value': str(trade['quantity']), 'inline': True},
                {'name': 'Order ID', 'value': trade['order_id'], 'inline': True}
            ]
        }
        
        return {
            'telegram': telegram_message,
            'discord': discord_embed
        }
    
    def format_position_update(self, position: Dict) -> Dict:
        """Format position update notification"""
        pnl_color = '🟢' if position['unrealized_pnl'] > 0 else '🔴' if position['unrealized_pnl'] < 0 else '⚪'
        
        telegram_message = f"""
<b>📊 POSITION UPDATE</b>

<b>Symbol:</b> {position['symbol']}
<b>Side:</b> {position['side']}
<b>Size:</b> {position['size']}
<b>Entry Price:</b> ${position['entry_price']:.2f}
<b>Current Price:</b> ${position['current_price']:.2f}
<b>P&L:</b> {pnl_color} ${position['unrealized_pnl']:.2f} ({position['unrealized_pnl_percent']:.2f}%)
<b>Leverage:</b> {position['leverage']}x
<b>Liquidation Price:</b> ${position['liquidation_price']:.2f}
"""
        
        discord_embed = {
            'title': f"📊 Position Update - {position['symbol']}",
            'color': 0x00ff00 if position['unrealized_pnl'] > 0 else 0xff0000 if position['unrealized_pnl'] < 0 else 0xffffff,
            'fields': [
                {'name': 'Side', 'value': position['side'], 'inline': True},
                {'name': 'Size', 'value': str(position['size']), 'inline': True},
                {'name': 'Entry Price', 'value': f"${position['entry_price']:.2f}", 'inline': True},
                {'name': 'Current Price', 'value': f"${position['current_price']:.2f}", 'inline': True},
                {'name': 'P&L', 'value': f"${position['unrealized_pnl']:.2f} ({position['unrealized_pnl_percent']:.2f}%)", 'inline': True},
                {'name': 'Leverage', 'value': f"{position['leverage']}x", 'inline': True}
            ]
        }
        
        return {
            'telegram': telegram_message,
            'discord': discord_embed
        }
    
    async def test_notifications(self) -> Dict:
        """Test all notification channels"""
        test_message = f"""
<b>🔔 TEST NOTIFICATION</b>

This is a test message from Quentrade.
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

All channels working properly!
"""
        
        test_embed = {
            'title': "🔔 Test Notification",
            'description': "This is a test message from Quentrade.",
            'color': 0x3498db,
            'fields': [
                {'name': 'Status', 'value': 'Test successful', 'inline': True},
                {'name': 'Time', 'value': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'inline': True}
            ]
        }
        
        results = {
            'telegram': False,
            'discord': False,
            'email': False
        }
        
        # Test Telegram
        if self.telegram_bot:
            try:
                results['telegram'] = await self.send_telegram_message(test_message)
            except Exception as e:
                self.logger.error(f"Telegram test failed: {e}")
        
        # Test Discord
        if self.discord_webhook_url:
            try:
                results['discord'] = self.send_discord_message("", embed=test_embed)
            except Exception as e:
                self.logger.error(f"Discord test failed: {e}")
        
        # Test Email
        if self.email_enabled:
            try:
                results['email'] = self.send_email(
                    "Quentrade Test Notification",
                    "<h2>Test Notification</h2><p>This is a test email from Quentrade.</p>",
                    html=True
                )
            except Exception as e:
                self.logger.error(f"Email test failed: {e}")
        
        return results

# Example usage
if __name__ == "__main__":
    notifier = QuentradeNotifications()
    
    # Test notifications
    async def test():
        results = await notifier.test_notifications()
        print("Notification test results:", results)
    
    asyncio.run(test())