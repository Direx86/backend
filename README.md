# Smart Home Energy Prediction - Backend API

Backend API untuk Prediksi Konsumsi Energi Rumah Tangga menggunakan metode Random Forest, Gradient Boosting, dan LSTM pada Sistem Smart Home.

**API:** [web-production-f2ae3.up.railway.app](https://web-production-f2ae3.up.railway.app)
**Swagger Docs:** [web-production-f2ae3.up.railway.app/docs](https://web-production-f2ae3.up.railway.app/docs)

## Tech Stack

| Teknologi | Versi | Keterangan |
|-----------|-------|------------|
| Python | 3.11 | Runtime |
| FastAPI | 0.115 | Web framework |
| Uvicorn | 0.34 | ASGI server |
| scikit-learn | 1.6.1 | Random Forest & Gradient Boosting |
| TensorFlow CPU | 2.16 | LSTM model |
| Pandas | 2.2 | Data processing |
| NumPy | 1.26 | Numerical computing |
| Joblib | 1.4 | Model serialization |

## Dataset

**Sumber:** UCI Machine Learning Repository - Individual Household Electric Power Consumption

| Detail | Keterangan |
|--------|------------|
| File | `household_power_consumption.txt` |
| Jumlah Baris | 2.075.259 (per menit) |
| Periode | Desember 2006 - November 2010 |
| Resampling | Rata-rata per jam -> 34.168 baris |
| Split | Train 70% (23.902) / Val 15% (5.121) / Test 15% (5.121) |

### Variabel

| Variabel | Unit | Deskripsi |
|----------|------|-----------|
| Global_active_power | kW | Total daya aktif (TARGET) |
| Global_reactive_power | kW | Total daya reaktif |
| Voltage | Volt | Tegangan listrik rata-rata |
| Global_intensity | Ampere | Arus listrik rata-rata |
| Sub_metering_1 | Wh | Dapur: dishwasher, oven, microwave |
| Sub_metering_2 | Wh | Laundry: mesin cuci, pengering, kulkas |
| Sub_metering_3 | Wh | Pemanas air listrik & AC |

## Model Machine Learning

### 1. Random Forest (TUNED)
- **Tipe:** Ensemble (Bagging)
- **Input:** 24 fitur (6 sensor + 4 kalender + 6 lag + 8 rolling)
- **File:** `random_forest_regressor_TUNED.joblib` (48 MB)
- **Metrik:** MAE=0.0166 | RMSE=0.0346 | R²=0.9988

### 2. Gradient Boosting (TUNED)
- **Tipe:** Ensemble (Boosting)
- **Input:** 24 fitur (6 sensor + 4 kalender + 6 lag + 8 rolling)
- **File:** `gradient_boosting_regressor_TUNED.joblib` (322 KB)
- **Metrik:** MAE=0.0158 | RMSE=0.0331 | R²=0.9989

### 3. LSTM
- **Tipe:** Deep Learning (RNN)
- **Input:** Window 24 jam Global_active_power (StandardScaler: mean=1.086397, std=0.929282)
- **Arsitektur:** LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dense(16) -> Dense(1)
- **File:** `lstm_model.keras` (389 KB)
- **Metrik:** MAE=0.3462 | RMSE=0.6883 | R²=0.5243

### Feature Engineering (24 fitur untuk Tree Models)

| Kategori | Jumlah | Fitur |
|----------|--------|-------|
| Sensor | 6 | Global_reactive_power, Voltage, Global_intensity, Sub_metering_1/2/3 |
| Kalender | 4 | hour, dayofweek, is_weekend, month |
| Lag | 6 | lag1, lag2, lag3, lag6, lag12, lag24 |
| Rolling | 8 | rollmean3, rollstd3, rollmean6, rollstd6, rollmean12, rollstd12, rollmean24, rollstd24 |

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/api/health` | Status server dan model yang ter-load |
| GET | `/api/metrics` | Metrik evaluasi (MAE, RMSE, R²) dari CSV |
| GET | `/api/models/info` | Informasi arsitektur dan parameter tiap model |
| GET | `/api/sample-data` | Data 168 jam (7 hari terakhir test set) + metrik |
| GET | `/api/full-test-data` | Data 5.121 points (full test set), query `?last_n=720` |
| GET | `/api/dataset-info` | Informasi dataset (jumlah baris, periode, variabel) |
| POST | `/api/predict` | Prediksi real-time dari input sensor (min 25 data point) |

### Contoh Response

**GET `/api/health`**
```json
{
  "status": "ok",
  "models_loaded": true,
  "rf": true,
  "gb": true,
  "lstm": true
}
```

**GET `/api/sample-data`** (ringkasan)
```json
{
  "data": [
    {
      "datetime": "2010-08-25T00:00:00",
      "actual": 0.536,
      "rf_pred": 0.534,
      "gb_pred": 0.533,
      "lstm_pred": 0.612
    }
  ],
  "metrics": [
    { "model": "RandomForest", "MAE": 0.0166, "RMSE": 0.0346, "R2": 0.9988 }
  ]
}
```

## Struktur Folder

```
backend/
├── app/
│   ├── main.py             # FastAPI entry point + semua endpoints
│   ├── config.py           # Konfigurasi model, scaler, fitur
│   ├── predictor.py        # Logic load model & prediksi
│   └── schemas.py          # Pydantic models (request/response)
├── models/
│   ├── random_forest_regressor_TUNED.joblib   # 48 MB
│   ├── gradient_boosting_regressor_TUNED.joblib  # 322 KB
│   ├── lstm_model.keras                       # 389 KB
│   └── test_metrics_default.csv               # Metrik evaluasi
├── data/
│   ├── sample_data.json        # 168 data points (7 hari)
│   ├── full_test_data.json     # 5.121 data points (full test set)
│   └── scaler_params.json      # Parameter scaler LSTM
├── generate_sample_data.py     # Script proses dataset UCI -> JSON
├── requirements.txt
├── Procfile                    # Railway start command
├── railway.json                # Railway build config
├── nixpacks.toml               # Pin Python 3.11
├── runtime.txt                 # Python version
└── .gitignore
```

## Setup Lokal

### Prasyarat
- Python 3.11+
- File model dari Google Colab:
  - `random_forest_regressor_TUNED.joblib`
  - `gradient_boosting_regressor_TUNED.joblib`
  - `lstm_model.keras`
  - `test_metrics_default.csv`

### Instalasi

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Generate Data dari Dataset UCI

Letakkan `household_power_consumption.txt` di folder `models/`, lalu jalankan:

```bash
python generate_sample_data.py
```

Script ini memproses dataset UCI asli -> menghasilkan `data/sample_data.json` dan `data/full_test_data.json`.

> **Catatan:** `household_power_consumption.txt` (127 MB) sudah di `.gitignore` karena melebihi batas GitHub 100 MB. File ini hanya dibutuhkan lokal.

### Jalankan Server

```bash
uvicorn app.main:app --reload --port 8000
```

- Server: `http://localhost:8000`
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

| Variable | Default | Keterangan |
|----------|---------|------------|
| `FRONTEND_URL` | `https://prediksi-konsumsi-energi-rumah-tangga.vercel.app` | URL frontend untuk CORS |
| `SCALER_MEAN` | `1.086397` | Mean StandardScaler LSTM |
| `SCALER_STD` | `0.929282` | Std StandardScaler LSTM |
| `PORT` | `8000` | Port server |

## Deploy ke Railway

1. Push repository ke GitHub
2. Buka [railway.app](https://railway.app) -> **New Project** -> **Deploy from GitHub Repo**
3. Railway otomatis mendeteksi `Procfile`, `railway.json`, dan `nixpacks.toml`
4. Set environment variable `FRONTEND_URL` ke URL Vercel frontend
5. Tunggu build & deploy selesai
6. Verifikasi: buka `https://[railway-url]/api/health`

### File Deploy

| File | Fungsi |
|------|--------|
| `Procfile` | Start command: `uvicorn app.main:app --host 0.0.0.0 --port ${PORT}` |
| `railway.json` | Build config (Nixpacks builder) |
| `nixpacks.toml` | Pin Python 3.11 |
| `runtime.txt` | Python version fallback |
| `requirements.txt` | Dependencies (menggunakan `tensorflow-cpu` untuk hemat RAM) |

### Troubleshooting Railway

| Masalah | Solusi |
|---------|--------|
| Model tidak ter-load | Cek logs. Pastikan file `.joblib`/`.keras` ada di repo |
| Memory exceeded | Pastikan menggunakan `tensorflow-cpu` bukan `tensorflow` |
| CORS error di frontend | Set `FRONTEND_URL` ke URL Vercel (tanpa trailing slash) |
| Build gagal | Pastikan `requirements.txt` ada di root folder |

## Skripsi

Proyek ini adalah bagian dari skripsi:
**"Prediksi Konsumsi Energi Rumah Tangga Menggunakan Metode Random Forest, Gradient Boosting, LSTM pada Sistem Smart Home"**

- **Nama:** Franscen Yosafat Sinambela
- **NIM:** 2244053
- **Program Studi:** Teknik Informatika
- **Kampus:** STMIK TIME, Medan
