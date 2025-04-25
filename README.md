# Quentrade - AI Machine Learning Trading Terminal

![Quentrade Banner](https://via.placeholder.com/800x200?text=Quentrade+AI+Trading+Terminal)

Quentrade adalah terminal trading cryptocurrency berbasis AI yang canggih, dirancang untuk berjalan di CLI Linux. Terminal ini menggabungkan analisis fundamental, sentimen pasar, data on-chain, dan indikator ekonomi makro untuk menghasilkan sinyal trading yang akurat.

## 🚀 Fitur Utama

- **AI Self-Thinking Engine**: Menganalisis dan membuat keputusan trading secara otonom
- **Multi-Source Data Integration**: Mengintegrasikan data dari Bybit, CoinMarketCap, FRED, News API, dan QuikNode
- **Advanced Sentiment Analysis**: Menganalisis sentimen dari 15+ sumber berita crypto utama
- **Real-Time Market Analysis**: Monitoring pasar 24/7 dengan analisis mendalam
- **Risk Management AI**: Manajemen risiko adaptif berdasarkan kondisi pasar
- **Strategy Studio**: Buat dan kelola strategi trading Anda sendiri
- **Backtest Engine**: Uji strategi pada data historis
- **Interactive CLI Interface**: Antarmuka yang mudah digunakan dengan visual yang menarik

## 📋 Prerequisites

- Linux OS (Ubuntu 20.04+ recommended)
- Python 3.8 atau lebih tinggi
- 2GB RAM minimum (4GB recommended)
- Koneksi internet stabil
- API Keys dari:
  - Bybit Exchange
  - CoinMarketCap
  - FRED (Federal Reserve Economic Data)
  - News API
  - QuikNode

## 🛠️ Instalasi

1. Clone repository:
```bash
https://github.com/Vixsry/quentrade-ai-trading.git
cd quentrade
```

2. Jalankan script setup:
```bash
chmod +x setup.sh
./setup.sh
```

3. Konfigurasi API keys di file `.env`:
```bash
nano .env
```

4. Aktivasi virtual environment:
```bash
source venv/bin/activate
```

5. Jalankan Quentrade:
```bash
python quentrade.py
```

## 📖 Cara Penggunaan

### Menu Utama

1. **SCAN COIN**: Analisis mendalam untuk coin tertentu
2. **NEWS SCAN**: Melihat berita crypto terkini dengan analisis sentimen
3. **CPI DATA**: Data ekonomi makro terbaru dan analisisnya
4. **SIGNAL TRADING**: Sinyal trading AI untuk beberapa coin
5. **STRATEGY STUDIO**: Buat dan kelola strategi trading
6. **BACKTEST ENGINE**: Uji strategi pada data historis
7. **SETTINGS**: Konfigurasi dan API keys
8. **LOG & STATS**: Statistik performa trading

### Contoh Penggunaan

#### Menganalisis Bitcoin:
1. Pilih menu [1] SCAN COIN
2. Masukkan "BTCUSDT"
3. Tunggu analisis AI
4. Lihat hasil analisis termasuk entry, TP, SL, dan confidence level

#### Membuat Strategi Baru:
1. Pilih menu [5] STRATEGY STUDIO
2. Pilih "Create New Strategy"
3. Masukkan nama dan parameter strategi
4. Simpan dan aktifkan strategi

## 🔧 Konfigurasi

### File .env
```env
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
COINMARKETCAP_API_KEY=your_api_key
FRED_API_KEY=your_api_key
NEWS_API_KEY=your_api_key
QUIKNODE_API_KEY=your_api_key
```

### Trading Settings
Edit pengaturan trading di `.env`:
```env
RISK_PER_TRADE=0.02  # Risiko per trade (2%)
MAX_POSITION_SIZE=0.1  # Maksimum ukuran posisi (10%)
USE_TESTNET=false  # Gunakan testnet untuk testing
```

## 📊 Struktur Database

Quentrade menggunakan SQLite untuk menyimpan:
- Histori sinyal trading
- Arsip berita dan analisis sentimen
- Strategi trading
- Log performa

## 🧠 AI Models

Quentrade menggunakan beberapa model AI:
- LSTM untuk prediksi pergerakan harga
- RandomForest untuk analisis sentimen
- GradientBoosting untuk assessment risiko
- RandomForest untuk pemilihan strategi

## 🔒 Keamanan

- API keys disimpan secara aman di file `.env`
- Tidak ada penyimpanan password dalam plaintext
- Semua komunikasi menggunakan HTTPS
- Regular security updates

## 🐛 Troubleshooting

### Masalah Umum

1. **API Connection Error**
   - Periksa koneksi internet
   - Verifikasi API keys di `.env`
   - Cek status layanan API

2. **Model Loading Error**
   - Hapus folder `models/` dan restart aplikasi
   - Sistem akan membuat model baru secara otomatis

3. **Database Error**
   - Jalankan: `python -c "from quentrade import QuentradeAI; app = QuentradeAI(); app.initialize_database()"`

## 📈 Performance Tips

1. Gunakan VPS dengan latensi rendah ke Bybit
2. Jalankan di server dedicated untuk performa optimal
3. Update model AI secara berkala dengan data terbaru
4. Monitor penggunaan API rate limits

## 🤝 Kontribusi

Kontribusi selalu diterima! Silakan buat pull request atau laporkan issues.

## 📄 Lisensi

MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## ⚠️ Disclaimer

Quentrade adalah alat bantu trading dan bukan jaminan profit. Selalu lakukan riset Anda sendiri dan gunakan manajemen risiko yang tepat. Trading cryptocurrency memiliki risiko tinggi dan bisa mengakibatkan kerugian finansial.

## 📞 Support

- Email: support@quentrade.ai
- Discord: [Join our community](https://discord.gg/quentrade)
- Telegram: [@quentrade_support](https://t.me/quentrade_support)

---

Dibuat dengan ❤️ oleh Tim Quentrade