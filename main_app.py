import streamlit as st
import os 
import pandas as pd
import pandas_ta # <-- ✅ Cukup impor library-nya, tidak perlu alias 'ta'
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import ML libraries
try:
    from xgboost import XGBRegressor
    from sqlalchemy import create_engine
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor #
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import VotingRegressor
    #from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Machine Learning libraries not available. Install scikit-learn for advanced predictions.")

# Page configuration
st.set_page_config(
    page_title="☕ Coffee Prediction System",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    body, .stApp {
        font-family: 'Poppins', sans-serif;
        color: #444;
    }

    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #8B4513 0%, #D2691E 50%, #CD853F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        animation: fadeInDown 1.2s ease-in-out;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #8B4513;
        margin: 0.8rem 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        color: #333;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }

    .metric-card strong {
        color: #8B4513;
        font-weight: 600;
    }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #fff;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }

    .prediction-card:hover {
        transform: scale(1.03);
    }

    .success-card {
        background: linear-gradient(135deg, #28a745 0%, #218838 100%);
        color: #ffffff;
        border-left-color: #28a745;
    }

    .warning-card {
        background: linear-gradient(135deg, #ffc107 0%, #e0a800 100%);
        color: #ffffff;
        border-left-color: #ffc107;
    }

    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

            
    footer, .footer {
    font-size: 0.8rem;
    color: #aaa;
    text-align: center;
    margin-top: 50px;
    }
    .stSlider > div > div {
    background-color: #f3f3f3;
    border-radius: 0.5rem;
    padding: 1rem;
    }

</style>
""", unsafe_allow_html=True)

class CoffeePredictionSystem:
    def __init__(self):
        # Mengambil URL database dari Environment Variable yang Anda set di Render
        self.db_url = os.getenv('DATABASE_URL')
        self.table_name = "data_kopi_export_bersih"  # Nama tabel baru di PostgreSQL
        # PERBAIKAN: Definisikan nama kolom database dalam urutan yang TEPAT
        self.db_column_order = [
            'tanggal', 'harga_penutupan', 'harga_pembukaan', 'harga_tertinggi',
            'harga_terendah', 'volume', 'perubahan_persen', 'perubahan_harga',
            'hari_minggu', 'range_harian', 'momentum', 'ma3', 'ma7',
            'google_trend', 'synthesized'
        ]
        if not self.db_url:
            # Pesan ini akan muncul jika Anda lupa mengatur Environment Variable
            st.error("FATAL: DATABASE_URL environment variable not set.")
            return
        
        # Mengubah URL 'postgres://' menjadi 'postgresql://' agar kompatibel
        if self.db_url.startswith("postgres://"):
            self.db_url = self.db_url.replace("postgres://", "postgresql://", 1)
            
        try:
            self.engine = create_engine(self.db_url)
        except Exception as e:
            st.error(f"Failed to create database engine: {e}")
            self.engine = None     

    def load_data(self):
        """Memuat data dari database PostgreSQL di Render."""
        
        if not self.engine:
            st.error("Database engine not initialized.")
            return None

        try:
            # Gunakan engine SQLAlchemy untuk membaca data
            query = f'SELECT * FROM "{self.table_name}"'
            df = pd.read_sql_query(query, self.engine)
            
            # Clean and process data
            df = self.clean_data(df)
            return df
            
        except Exception as e:
            st.error(f"Error loading data from PostgreSQL: {e}")
            return None
    
    # GANTI FUNGSI clean_data LAMA ANDA DENGAN YANG INI

    def clean_data(self, df):
        """Clean, validate, and process the raw data with outlier detection."""
        st.toast("Starting data cleaning process...", icon="🧹")

        # 1. VALIDASI STRUKTUR & PENAMAAN KOLOM
        # =======================================
        expected_cols = 15 # Jumlah kolom yang diharapkan
        if df.shape[1] != expected_cols:
            st.error(f"Data Loading Error: Expected {expected_cols} columns, but found {df.shape[1]}. Please check DB schema.")
            st.write("--- END Debugging clean_data (ERROR) ---")
            return None

        # Ini penting jika data DB Anda masih memiliki header di baris pertama
        if not df.empty and df.iloc[0, 0] == 'Tanggal':
            st.warning("Removing header row 'Tanggal'.")
            df = df.iloc[1:].reset_index(drop=True)
            st.write(f"DF shape after removing header: {df.shape}")
    
        #column_names = [
        #    'tanggal', 'harga_penutupan', 'harga_pembukaan', 'harga_tertinggi', 
        #   'harga_terendah', 'volume', 'perubahan_persen', 'perubahan_harga',
        #   'hari_minggu', 'range_harian', 'momentum', 'ma3', 'ma7', 
        #   'google_trend', 'synthesized'
        #]
        # Pastikan jumlah kolom sama sebelum assignment


    
        # 2. KONVERSI TIPE DATA
        # ========================
        df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
        numeric_cols = [
            'harga_penutupan', 'harga_pembukaan', 'harga_tertinggi', 'harga_terendah', 
            'volume', 'perubahan_persen', 'perubahan_harga', 'range_harian', 
            'momentum', 'ma3', 'ma7', 'google_trend'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
        df['synthesized'] = df['synthesized'].map({'TRUE': True, 'FALSE': False, True: True, False: False}).fillna(False)
    


        # 3. PEMBERSIHAN AWAL & PENGURUTAN
        # =================================
        # Hapus baris jika data tanggal atau harga penutupan tidak valid
        df.dropna(subset=['tanggal', 'harga_penutupan'], inplace=True)
        df.sort_values('tanggal', inplace=True)
        df.reset_index(drop=True, inplace=True)
    

        
        # 4. PENGISIAN NILAI KOSONG (SMART FILL)
        # =======================================
        missing_before = df[numeric_cols].isnull().sum().sum()
        if missing_before > 0:
        # Gunakan forward fill, cocok untuk time-series (mengambil nilai valid terakhir)
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        # Isi sisa NaN di awal data dengan back fill
            df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
            st.toast(f"✨ Filled {missing_before} missing data point(s) using ffill/bfill.", icon="🪄")
        st.write(f"Number of NaNs after fillna: {df.isnull().sum().sum()}")

        # Sebagai fallback terakhir untuk NaN yang masih tersisa di kolom numerik
        for col in numeric_cols:
            if df[col].isnull().any():
                # Hitung median. Jika kolom seluruhnya NaN, median() akan mengembalikan NaN.
                calculated_median = df[col].median()
                # Tentukan nilai pengisi: 0 jika calculated_median adalah NaN, jika tidak gunakan calculated_median
                fill_value = 0 if pd.isna(calculated_median) else calculated_median
                df[col] = df[col].fillna(fill_value)
                st.warning(f"⚠️ Kolom '{col}' masih memiliki NaN setelah ffill/bfill, diisi dengan {fill_value}.")

        # 5. DETEKSI & HAPUS OUTLIER (Z-SCORE)
        # =======================================
        initial_rows = len(df)
        # Pastikan hanya kolom numerik yang ada yang digunakan
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        #Cek apakah ada cukup data untuk Z-score (min 2 data points)
        if len(df) > 1 and not df[existing_numeric_cols].empty:
             # Hitung Z-score pada data numerik
            z_scores = np.abs(stats.zscore(df[existing_numeric_cols]))
            # Buat filter untuk data yang bukan outlier (Z-score < 3)
            non_outlier_mask = (z_scores < 3).all(axis=1)
            df = df[non_outlier_mask]
        else:
            st.info("Tidak cukup data untuk melakukan deteksi outlier atau tidak ada kolom numerik yang relevan.")
            
        outliers_removed = initial_rows - len(df)
        if outliers_removed > 0:
            st.warning(f"⚠️ Removed {outliers_removed} outlier row(s) to improve data quality.")
        st.write(f"DF shape after outlier removal: {df.shape}")
        st.write("DF TAIL after outlier removal (final check):")
        st.dataframe(df.tail())

        st.toast("Data cleaning complete!", icon="✅")
        st.write("--- END Debugging clean_data ---")
        return df   
    
    def add_new_data(self, new_data_dict):
        """Menambahkan data baru ke database PostgreSQL di Render."""
        if not self.engine:
            st.error("Database engine not initialized.")
            return False
        
        try:
            df_new = pd.DataFrame([new_data_dict])
                # PERBAIKAN: Susun ulang kolom df_new agar sesuai dengan urutan kolom di database
            # yang telah Anda definisikan di self.db_column_order
            # Pastikan self.db_column_order sudah didefinisikan di __init__
            if hasattr(self, 'db_column_order'):
                df_new = df_new[self.db_column_order]
            else:
                st.error("Internal Error: db_column_order not defined in CoffeePredictionSystem.")
                return False

            df_new.to_sql(self.table_name, self.engine, if_exists='append', index=False)
            return True
            
        except Exception as e:
            st.error(f"Error adding data to PostgreSQL: {e}")
            # Cetak error lebih detail di konsol untuk debugging
            print(f"DETAIL ERROR adding data: {e}")
            return False
    
    def get_statistics(self, df):
        """Get data statistics"""
        if df is None or df.empty:
            return {}
        
        stats = {
            'total_records': len(df),
            'date_range': f"{df['tanggal'].min().strftime('%Y-%m-%d')} to {df['tanggal'].max().strftime('%Y-%m-%d')}",
            'current_price': df['harga_penutupan'].iloc[-1],
            'avg_price': df['harga_penutupan'].mean(),
            'min_price': df['harga_penutupan'].min(),
            'max_price': df['harga_penutupan'].max(),
            'avg_volume': df['volume'].mean() if 'volume' in df.columns else 0,
            'price_change': df['harga_penutupan'].iloc[-1] - df['harga_penutupan'].iloc[-2] if len(df) > 1 else 0,
            'synthesized_count': df['synthesized'].sum() if 'synthesized' in df.columns else 0
        }
        
        return stats

                    # --- SEMUA FUNGSI LAINNYA DIMULAI DARI SINI (TANPA INDENTASI) ---

def create_features(df):
    """Membuat fitur-fitur untuk machine learning.
    Fungsi ini akan menghitung ulang fitur-fitur penting
    seperti perubahan_harga (arah), perubahan_persen, momentum,
    serta moving averages (ma3, ma7) berdasarkan seluruh data historis
    yang sudah dimuat dan dibersihkan. """
    features_df = df.copy()

    # Pastikan kolom 'tanggal' adalah tipe datetime dan DataFrame terurut berdasarkan tanggal.
    # Ini penting untuk perhitungan diff() dan shift() yang akurat.
    features_df['tanggal'] = pd.to_datetime(features_df['tanggal'])
    features_df = features_df.sort_values('tanggal').reset_index(drop=True)

    # ===================================================================
    # === PERBAIKAN & PENGHITUNGAN ULANG FITUR INTI ===
    # ===================================================================

    # 1. Hitung ulang 'perubahan_harga' sebagai indikator arah (1=naik, -1=turun, 2=sideways)
    # Ini akan menimpa kolom 'perubahan_harga' yang mungkin datang dari DB dengan nilai yang salah.
    # Perhitungan berdasarkan selisih harga penutupan hari ini dengan hari sebelumnya.
    price_diff_close_prev = features_df['harga_penutupan'].diff()
    conditions_direction = [
        price_diff_close_prev > 0,  # Kondisi untuk harga naik
        price_diff_close_prev < 0   # Kondisi untuk harga turun
    ]
    choices_direction = [1, -1]
    
    # Default adalah 2 (sideways) untuk kasus di mana selisihnya 0 atau NaN (baris pertama)
    features_df['perubahan_harga'] = np.select(conditions_direction, choices_direction, default=2).astype(int)

    # 2. Hitung ulang 'perubahan_persen' (aktual: (harga_penutupan hari ini - harga_penutupan kemarin) / harga_penutupan kemarin)
    # Ini akan menimpa kolom 'perubahan_persen' yang mungkin datang dari DB dengan nilai yang salah.
    features_df['perubahan_persen'] = (features_df['harga_penutupan'].diff() / features_df['harga_penutupan'].shift(1))
    # Isi NaN di baris pertama (karena tidak ada data kemarin) dengan 0
    features_df['perubahan_persen'].fillna(0, inplace=True)

    # 3. Hitung ulang 'momentum' (menggunakan 'perubahan_persen' yang sudah benar)
    # Jika definisi momentum Anda sama dengan perubahan persen, gunakan ini.
    # Jika ada definisi lain, sesuaikan di sini.
    features_df['momentum'] = features_df['perubahan_persen']

    # ===============================================================
    # FITUR BARU & PENYESUAIAN 
    # ===============================================================

    # 1. Indikator Teknikal Lanjutan dari Pandas_ta

    # RSI (Relative Strength Index) - Mengukur momentum
    features_df.ta.rsi(length=14, append=True)
    
    # MACD (Moving Average Convergence Divergence) - Menunjukkan tren & momentum
    features_df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Bollinger Bands - Mengukur volatilitas
    features_df.ta.bbands(length=20, append=True)

    # 2. Technical indicators
    features_df['sma_5'] = features_df['harga_penutupan'].rolling(5).mean()
    features_df['sma_10'] = features_df['harga_penutupan'].rolling(10).mean()
    features_df['sma_20'] = features_df['harga_penutupan'].rolling(20).mean()
    
    # 3. Fitur Tren (Slope) dan Indeks Volatilitas
    # Pastikan ma3 dan ma7 ada, jika tidak, hitung dari rolling mean
    if 'ma3' not in features_df.columns:
        features_df['ma3'] = features_df['rolling_mean_3']
    if 'ma7' not in features_df.columns:
        features_df['ma7'] = features_df['rolling_mean_7']
    features_df['slope_ma'] = features_df['ma3'] - features_df['ma7']
    features_df['volatility_index'] = (features_df['harga_tertinggi'] - features_df['harga_terendah']) / features_df['harga_penutupan']
    
    # 3. Price ratios
    features_df['high_low_ratio'] = features_df['harga_tertinggi'] / features_df['harga_terendah']
    features_df['close_open_ratio'] = features_df['harga_penutupan'] / features_df['harga_pembukaan']
    
    # 4. Volatility
    features_df['volatility'] = features_df['harga_penutupan'].rolling(10).std()
    
    # 1. Fitur berbasis Waktu
    features_df['day_of_week'] = features_df['tanggal'].dt.dayofweek  # Senin=0, Minggu=6
    features_df['is_weekend'] = (features_df['tanggal'].dt.dayofweek >= 5).astype(int) # Sabtu/Minggu = 1
    features_df['day_of_year'] = features_df['tanggal'].dt.dayofyear
    features_df['week_of_year'] = features_df['tanggal'].dt.isocalendar().week.astype(int)
    features_df['month'] = features_df['tanggal'].dt.month

    # 6. Lag features
    for lag in [1, 3, 7, 14]:
        features_df[f'price_lag_{lag}'] = features_df['harga_penutupan'].shift(lag)
        features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)
    
     # 2. Fitur Rolling Mean & Volatilitas (Rolling Std)
    features_df['rolling_mean_3'] = features_df['harga_penutupan'].rolling(window=3).mean()
    features_df['rolling_mean_7'] = features_df['harga_penutupan'].rolling(window=7).mean()
    features_df['rolling_mean_14'] = features_df['harga_penutupan'].rolling(window=14).mean()
    features_df['rolling_std_7'] = features_df['harga_penutupan'].rolling(window=7).std()

    # 4. Exponential Moving Averages (EMA)
    features_df['ema_5'] = features_df['harga_penutupan'].ewm(span=5, adjust=False).mean()
    features_df['ema_12'] = features_df['harga_penutupan'].ewm(span=12, adjust=False).mean()
    
    # 7. Trend features
    features_df['price_trend_5'] = features_df['harga_penutupan'].rolling(5).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
    features_df['price_trend_10'] = features_df['harga_penutupan'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else 0)
    
    # 8. Hapus semua baris yang mengandung NaN setelah membuat fitur
    return features_df.dropna()

# Fungsi untuk menghitung semua metrik
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Hindari pembagian dengan nol jika y_true ada yang 0
    y_true_safe = y_true.replace(0, 1e-6)
    mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    return {"MAE": mae, "MSE": mse, "R²": r2, "MAPE (%)": mape}

# Fungsi untuk membuat grafik prediksi vs aktual
def plot_prediction_vs_actual(y_true, y_pred, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Prediksi',
                             marker=dict(color='rgba(100, 149, 237, 0.7)')))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
                             mode='lines', name='Garis Ideal (x=y)', line=dict(dash='dash', color='red')))
    fig.update_layout(title=f'Prediksi vs. Aktual ({model_name})',
                      xaxis_title='Harga Aktual', yaxis_title='Harga Prediksi')
    return fig

# Fungsi untuk membuat grafik residual
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals',
                             marker=dict(color='rgba(47, 79, 79, 0.7)')))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(title=f'Grafik Residual ({model_name})',
                      xaxis_title='Harga Prediksi', yaxis_title='Error (Aktual - Prediksi)')
    return fig

# Fungsi untuk membuat grafik feature importance
def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    df_importance = pd.DataFrame({'Fitur': feature_names, 'Pentingnya': importances})
    df_importance = df_importance.sort_values(by='Pentingnya', ascending=False).head(15) # Ambil 15 teratas
    
    fig = px.bar(df_importance, x='Pentingnya', y='Fitur', orientation='h',
                 title='15 Fitur Paling Penting')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def train_and_predict_single_model(df, model_name, days_ahead):
    """Trains a single chosen model on all data and predicts the future."""
    if not ML_AVAILABLE:
        st.warning("ML libraries not available.")
        return simple_prediction(df, days_ahead)

    st.info(f"Menggunakan model terpilih: **{model_name}** untuk prediksi.")

    # Siapkan data lengkap (tanpa split)
    features_df = create_features(df)
    feature_cols = [col for col in features_df.columns if col not in ['tanggal', 'harga_penutupan', 'perubahan_harga', 'synthesized']]
    X = features_df[feature_cols].fillna(0)
    y = features_df['harga_penutupan']
    
    # Konfigurasi model
    model_config = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1),
        'CatBoost': CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    }
    model = model_config.get(model_name, LGBMRegressor()) # Default ke LGBM jika nama tidak ditemukan

    # Penskalaan fitur pada seluruh data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Latih pada 100% data yang sudah diskalakan
    model.fit(X_scaled, y)

    # Dapatkan baris fitur terakhir untuk membuat prediksi
    last_features_scaled = scaler.transform(X.tail(1))
    
    # Prediksi untuk 1 hari ke depan sebagai titik referensi
    # (Metode ini lebih sederhana dan cepat daripada melatih ulang model direct)
    next_day_pred = model.predict(last_features_scaled)[0]
    
    # Asumsikan tren linear sederhana dari harga terakhir ke prediksi hari berikutnya
    current_price = df['harga_penutupan'].iloc[-1]
    trend = next_day_pred - current_price
    
    # Buat prediksi untuk beberapa hari ke depan dengan ekstrapolasi tren
    predictions = [current_price + trend * i for i in range(1, days_ahead + 1)]

    return np.array(predictions)

def evaluate_models_with_tscv(df):
    """Evaluates models using TimeSeriesSplit for robust performance metrics."""
    if not ML_AVAILABLE:
        st.warning("Machine Learning libraries not available.")
        return None

    # Persiapan data
    features_df = create_features(df)
    feature_cols = [col for col in features_df.columns if col not in ['tanggal', 'harga_penutupan', 'perubahan_harga', 'synthesized']]
    X = features_df[feature_cols].fillna(0)
    y = features_df['harga_penutupan']

    # Model dengan hyperparameter terbaik yang sudah kita tentukan
    models = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1, eval_metric='mae'),
        'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1),
        'CatBoost': CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
    }

    # Siapkan TimeSeriesSplit, 3 splits cukup untuk kecepatan di app
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    
    # Penskalaan fitur
    scaler = StandardScaler()

    # Mulai evaluasi
    for name, model in models.items():
        mae_scores = []
        progress_bar = st.progress(0, text=f"Menguji {name}...")

        # Loop melalui setiap fold cross-validation
        for i, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Skalakan data di setiap fold untuk menghindari data leakage
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, predictions)
            mae_scores.append(mae)
            progress_bar.progress((i + 1) / tscv.n_splits, text=f"Menguji {name} - Fold {i+1}/{tscv.n_splits}")

        # Simpan hasil rata-rata dan standar deviasi
        results.append({
            'Model': name,
            'Rata-rata MAE': np.mean(mae_scores),
            'Std Dev MAE': np.std(mae_scores)
        })
        progress_bar.empty()

    return pd.DataFrame(results)




def train_ml_models(df):
    """Train multiple ML models"""
    if not ML_AVAILABLE:
        return None, None, {}
    
    # 1. Persiapan Fitur dan Data
    # ============================
    # Create features
    features_df = create_features(df)
    
    # Select feature columns
    feature_cols = [col for col in features_df.columns 
                   if col not in ['tanggal', 'harga_penutupan', 'perubahan_harga', 'synthesized']]
    
    X = features_df[feature_cols].fillna(0)
    y = features_df['harga_penutupan']
    
    # Pembagian data berbasis waktu (Time-series split)
    test_size = 0.2
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Pastikan tidak ada pengacakan (shuffling)
    st.info(f"Data latih: {X_train.shape[0]} baris, Data uji: {X_test.shape[0]} baris.")


    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Definisi Model yang Komprehensif
    # ==================================
    # Train models
    models = {
        # Model dasar sebagai baseline
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        
        # Model Gradient Boosting dengan hyperparameter yang lebih baik
        'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4,
        subsample=0.8, # Mencegah overfitting dengan mengambil 80% data per pohon 
        random_state=42),
        'XGBoost': XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8, # Mengambil 80% fitur per pohon
        random_state=42,
        n_jobs=-1,
        eval_metric='mae'),
        'LightGBM': LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31, # Parameter utama untuk kompleksitas, 31 adalah default yang baik
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1),
        'CatBoost': CatBoostRegressor(
        iterations=300, # Sama dengan n_estimators
        learning_rate=0.05,
        depth=6, # Mirip dengan max_depth
        l2_leaf_reg=3, # Parameter regulerisasi L2
        verbose=0,
        random_state=42)
    }
    
    
    # 3. Pelatihan dan Evaluasi Model
    # ===============================
    model_scores = {}
    best_model = None
    best_score = float('inf')
    
    for name, model in models.items():
        # Train model
        if name == 'Random Forest':
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        model_scores[name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
        
        # Pemilihan model terbaik berdasarkan MAE (Mean Absolute Error)
        if mae < best_score:
            best_score = mae
            best_model = (model, scaler if name != 'Random Forest' else None)
    
    return best_model, X.columns.tolist(), model_scores

def train_and_predict_ensemble(df, days_ahead):
    """Trains an ensemble of models on all data and predicts the future."""
    if not ML_AVAILABLE:
        st.warning("ML libraries not available.")
        return simple_prediction(df, days_ahead)

    # Persiapan data
    features_df = create_features(df)
    feature_cols = [col for col in features_df.columns if col not in ['tanggal', 'harga_penutupan', 'perubahan_harga', 'synthesized']]
    X = features_df[feature_cols].fillna(0)
    
    # Penskalaan fitur pada seluruh data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Definisikan model-model dasar
    base_models = [
        ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)),
        ('lgb', LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)),
        ('cat', CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42))
    ]
    
    # Buat model ensemble
    ensemble_model = VotingRegressor(estimators=base_models)

    # --- Gunakan strategi Direct Prediction untuk stabilitas ---
    # Buat target untuk model direct prediction
    target_df = features_df.copy()
    target_df['target'] = target_df['harga_penutupan'].shift(-days_ahead)
    target_df = target_df.dropna()

    X_direct_scaled = scaler.transform(target_df[feature_cols])
    y_direct = target_df['target']
    
    # Latih model ensemble pada tugas direct prediction
    ensemble_model.fit(X_direct_scaled, y_direct)
    
    # Dapatkan satu prediksi final untuk hari ke-N
    last_features_scaled = scaler.transform(X.tail(1))
    predicted_price_final = ensemble_model.predict(last_features_scaled)[0]
    
    # Interpolasi untuk visualisasi
    current_price = df['harga_penutupan'].iloc[-1]
    predictions = np.linspace(current_price, predicted_price_final, days_ahead)

    return predictions




def train_direct_model(df, days_to_predict=30):
    """
    Trains a model to predict the price 'days_to_predict' days into the future directly.
    This avoids iterative feedback loops.
    """
    if not ML_AVAILABLE:
        return None, None

    # 1. Buat fitur seperti biasa
    features_df = create_features(df)
    
    # 2. Buat target (y) dengan menggeser harga penutupan ke masa lalu
    # Ini adalah inti dari metode direct: prediksi harga N hari ke depan
    features_df['target'] = features_df['harga_penutupan'].shift(-days_to_predict)

    # 3. Buang baris yang tidak memiliki target (N baris terakhir) dan fitur NaN
    features_df = features_df.dropna()

    # 4. Siapkan data X dan y
    feature_cols = [col for col in features_df.columns if col not in ['tanggal', 'harga_penutupan', 'target', 'perubahan_harga', 'synthesized']]
    X = features_df[feature_cols]
    y = features_df['target']

    # 5. Latih model seperti biasa (hanya pakai satu model untuk kesederhanaan)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X, y)

    model_info = {'model': model, 'scaler': None} # Scaler tidak terlalu penting untuk RF
    
    return model_info, feature_cols

def make_direct_prediction(model_info, feature_cols, df):
    """
    Makes a single prediction for the future using the direct model.
    """
    if model_info is None:
        return None

    model = model_info['model']
    
    # Siapkan input: hanya butuh baris terakhir dari data historis
    features_df = create_features(df)
    last_features = features_df[feature_cols].iloc[-1:]

    # Lakukan prediksi satu kali
    predicted_price = model.predict(last_features)[0]
    
    return predicted_price

def make_predictions(model_info, feature_cols, df, days_ahead):
    """Make predictions using trained model"""
    if model_info is None:
        return simple_prediction(df, days_ahead)
    
    model, scaler = model_info
    
    # Get last row features
    features_df = create_features(df)
    last_features = features_df[feature_cols].iloc[-1:].fillna(0)
    
    predictions = []
    current_features = last_features.copy()
    
    for _ in range(days_ahead):
        if scaler is not None:
            pred_input = scaler.transform(current_features)
            pred = model.predict(pred_input)[0]
        else:
            pred = model.predict(current_features)[0]
        
        predictions.append(pred)
        
        # Update features for next prediction (simplified)
        current_features.iloc[0, 0] = pred  # Update first feature with prediction
    
    return np.array(predictions)

def simple_prediction(df, days_ahead):
    """Simple trend-based prediction"""
    recent_prices = df['harga_penutupan'].tail(30)
    
    # Calculate trend
    x = np.arange(len(recent_prices))
    trend = np.polyfit(x, recent_prices, 1)[0]
    
    # Base prediction
    base_price = recent_prices.mean()
    predictions = []
    
    for i in range(days_ahead):
        seasonal = 50 * np.sin(2 * np.pi * i / 365)  # Annual seasonality
        pred = base_price + (trend * i) + seasonal + np.random.normal(0, 10)
        predictions.append(max(pred, 100))  # Minimum price floor
    
    return np.array(predictions)


def create_charts(df, predictions=None, days_ahead=None):
    """Create visualization charts"""
    # Price history chart
    recent_data = df.tail(90)
    
    fig_history = go.Figure()
    
    # Candlestick chart
    fig_history.add_trace(go.Candlestick(
        x=recent_data['tanggal'],
        open=recent_data['harga_pembukaan'],
        high=recent_data['harga_tertinggi'],
        low=recent_data['harga_terendah'],
        close=recent_data['harga_penutupan'],
        name='Coffee Price'
    ))
    
    fig_history.update_layout(
        title="Coffee Price History (Last 90 Days)",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    # Prediction chart
    fig_prediction = None
    if predictions is not None and days_ahead is not None:
        fig_prediction = go.Figure()
        
        # Historical data
        recent_data = df.tail(60)
        fig_prediction.add_trace(go.Scatter(
            x=recent_data['tanggal'],
            y=recent_data['harga_penutupan'],
            mode='lines',
            name='Historical Price',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Future dates
        last_date = df['tanggal'].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(predictions),
            freq='D'
        )
        
        # Predictions
        fig_prediction.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=4)
        ))
        
        fig_prediction.update_layout(
            title=f"Coffee Price Prediction - Next {days_ahead} Days",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=500
        )
    
    return fig_history, fig_prediction

def main():
    # Header
    st.markdown('<h1 class="main-header">☕ Coffee Prediction System</h1>', unsafe_allow_html=True)
    
    # Inisialisasi session state (jika Anda ingin menggunakan fitur simpan model terbaik)
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = 'XGBoost' 
    
    # Initialize system
    coffee_system = CoffeePredictionSystem()
    
    # --- KODE UNTUK SIDEBAR APLIKASI ---
    with st.sidebar:
    # 1. BAGIAN HEADER & LOGO
    # Pastikan file "logo-PTSAG.png" ada di dalam repository GitHub Anda.
    # Render akan menemukannya secara otomatis saat deploy.
        st.image("logo-PTSAG.png", width=160)
        st.markdown("## ☕ Sistem Prediksi Harga Kopi")
        st.markdown("### ⚙️ Prediction Settings")
        st.divider() # Garis pemisah untuk tata letak yang bersih

    # 2. BAGIAN PENGATURAN PREDIKSI
        st.markdown("#### **Pilih Rentang Waktu Prediksi**")

    # Ambil tanggal hari ini sebagai titik awal
        today = datetime.now().date()
    
    # Atur tanggal prediksi default (30 hari dari sekarang)
        default_prediction_date = today + timedelta(days=30)

    # Widget untuk memilih tanggal akhir prediksi
        selected_date = st.date_input(
            label="Prediksi Hingga Tanggal:",
            value=default_prediction_date,
            min_value=today + timedelta(days=1),      # Prediksi minimal untuk besok
            max_value=today + timedelta(days=365*2),  # Batas prediksi hingga 2 tahun
            help="Pilih tanggal akhir untuk melihat hasil prediksi harga."
    )

    # 3. PERHITUNGAN DAN FEEDBACK UNTUK PENGGUNA
    # Logika ini sudah benar. Jika ada selisih, masalahnya bukan di sini.
    # Feedback yang jelas ini akan membantu Anda melacak masalahnya.
        if selected_date:
            days_ahead = (selected_date - today).days
        
        # Tampilkan informasi dengan jelas menggunakan st.info atau st.success
            st.info(
            f"Prediksi akan dibuat untuk **{days_ahead} hari** ke depan, "
            f"mulai dari **{today.strftime('%d %B %Y')}** hingga **{selected_date.strftime('%d %B %Y')}**."
        )
        else:
        # Fallback jika terjadi error (meskipun jarang dengan st.date_input)
            days_ahead = 0 
            st.error("Tanggal tidak valid. Silakan pilih kembali.")

    # Anda bisa menambahkan tombol untuk memicu prediksi jika perlu
    # predict_button = st.button("Jalankan Prediksi", type="primary")

            st.divider()
            st.caption("© 2025 - FM")
        
        prediction_method = st.selectbox(
            "Prediction Method:",
            # Urutkan dari yang paling andal ke yang paling sederhana
            ["Machine Learning (Ensemble)", "Machine Learning (Uji Silang)", "Machine Learning (Direct)", "Machine Learning (Iterative)", "Simple Trend"] if ML_AVAILABLE else ["Simple Trend"],
            help="""
            - **Machine Learning (Ensemble)**: Proses cepat. Menggabungkan 3 model terbaik.
            - **Machine Learning (Uji Silang)**: Proses Lambat. Menemukan model terbaik via cross-validation lalu membuat prediksi.
            - **Direct**: Cepat. Melatih beberapa model & memilih yang terbaik dari 1x uji.
            - **Iterative**: Cepat. Versi alternatif dari metode Direct.
            - **Simple Trend**: Sangat cepat, prediksi dasar.
            """
        )
        
        st.markdown("### 📊 Data Management")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # ... (kode expander untuk 'Add New Data') ...

        # Data addition section
        st.markdown("### ➕ Add New Data")
        with st.expander("Add Coffee Price Data"):
            new_date = st.date_input("Date", datetime.now())
            new_close = st.number_input("Closing Price", min_value=0.0, value=2000.0)
            new_open = st.number_input("Opening Price", min_value=0.0, value=2000.0)
            new_high = st.number_input("Highest Price", min_value=0.0, value=2050.0)
            new_low = st.number_input("Lowest Price", min_value=0.0, value=1950.0)
            new_volume = st.number_input("Volume (In Tons)", min_value=0.0, value=5000.0)
            new_google_trend = st.number_input("Google Trend Score (0-100)", min_value=0, max_value=100, 
            value=50  # Gunakan nilai tengah sebagai default
            )
            if st.button("💾 Add Data"):
                # Calculate derived values based on current input
                # Note: True price_change_pct and momentum (based on prev day)
                # will be calculated in clean_data/create_features after loading ALL data.
                
                # price_change ini masih new_close - new_open, bukan perubahan harian
                temp_price_change = new_close - new_open
                temp_price_change_pct = (temp_price_change / new_open) if new_open > 0 else 0

                daily_range = new_high - new_low
                day_of_week = new_date.weekday() + 1
                
                
                # Siapkan data dalam bentuk dictionary
                new_data_dict = {
                    'tanggal': new_date,
                    'harga_penutupan': new_close,
                    'harga_pembukaan': new_open,
                    'harga_tertinggi': new_high,
                    'harga_terendah': new_low,
                    'volume': new_volume / 1000, # <-- Volume dibagi 1000 untuk konversi ke kg
                    'perubahan_persen': temp_price_change_pct, # <-- Tanpa *100
                    'perubahan_harga': temp_price_change, # <-- Biarkan sebagai selisih open-close
                    'hari_minggu': day_of_week,
                    'range_harian': daily_range,
                    'momentum': temp_price_change_pct, # Placeholder (akan dihitung ulang di create_features)
                    'ma3': new_close, # Placeholder (akan dihitung ulang di create_features)
                    'ma7': new_close, # Placeholder (akan dihitung ulang di create_features)
                    'google_trend': new_google_trend,
                    'synthesized': False
                }
                
                if coffee_system.add_new_data(new_data_dict):
                    st.success("✅ Data added successfully!")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("❌ Failed to add data")
    
    # Load data
    with st.spinner("Loading coffee data..."):
        df = coffee_system.load_data()
        
        if df is None or df.empty:
            st.error("❌ Failed to load data")
            return
        
        stats = coffee_system.get_statistics(df)
    
    # Main dashboard
    st.markdown("### 📊 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f'''
        <div class="prediction-card">
            <h4>Current Price</h4>
            <h2>$ {stats['current_price']:,.0f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        change = stats['price_change']
        color = "#28a745" if change > 0 else "#dc3545"
        arrow = "↗️" if change > 0 else "↘️"
        st.markdown(f'''
        <div class="prediction-card">
            <h4>Price Change</h4>
            <h2 style="color: {color}">{arrow} $ {abs(change):,.0f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="prediction-card">
            <h4>Average Price</h4>
            <h2>$ {stats['avg_price']:,.0f}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="prediction-card">
            <h4>Total Records</h4>
            <h2>{stats['total_records']:,}</h2>
        </div>
        ''', unsafe_allow_html=True)
    
    # Tabs for different sections
    tab1, tab2, tab_eval, tab3, tab4 = st.tabs(["📈 Charts", "🔮 Predictions","🧪 Uji Model & Diagnostik", "📊 Statistics", "📋 Data"])
    
    with tab1:
        st.markdown("### 📈 Price Charts")
        
        # Create and display charts
        fig_history, _ = create_charts(df)
        st.plotly_chart(fig_history, use_container_width=True)
        
        # Volume chart
        if 'volume' in df.columns:
            recent_data = df.tail(90)
            fig_volume = px.bar(
                recent_data, 
                x='tanggal', 
                y='volume',
                title="Trading Volume (Last 90 Days)"
            )
            st.plotly_chart(fig_volume, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔮 Price Predictions")
        
        if st.button("🚀 Generate Predictions", type="primary"):
            with st.spinner("Training models and generating predictions..."):
                
                if prediction_method == "Machine Learning (Ensemble) " and ML_AVAILABLE:           
                    st.info("Proses lambat. Menggabungkan 3 model terbaik....")
                    predictions = train_and_predict_ensemble(df, days_ahead)
                    method_used = "ML Ensemble (Voting Regressor)"
                
                elif prediction_method == "ML (Uji Silang & Model Terbaik)" and ML_AVAILABLE:
                    st.info("Menjalankan proses evaluasi mendalam dengan Cross-Validation untuk menemukan model terbaik...")
            
                    # 1. Jalankan evaluasi yang ketat untuk menemukan model terbaik
                    evaluation_results_df = evaluate_models_with_tscv(df)
            
                    if evaluation_results_df is not None and not evaluation_results_df.empty:
                        # 2. Ambil nama model terbaik
                        best_performer = evaluation_results_df.sort_values(by='Rata-rata MAE').iloc[0]
                        best_model_name = best_performer['Model']
                        st.success(f"Model terbaik berdasarkan uji silang: **{best_model_name}**. Melatih ulang model ini pada semua data untuk prediksi akhir...")

                        # 3. Latih ulang HANYA model terbaik pada 100% data dan buat prediksi
                        # (Ini menggunakan logika dari train_and_predict_single_model yang kita bahas sebelumnya)
                        predictions = train_and_predict_single_model(df, best_model_name, days_ahead)
                        method_used = f"Uji Silang + {best_model_name}"
                    else:
                        st.error("Gagal melakukan evaluasi model. Beralih ke Simple Trend.")
                        predictions = simple_prediction(df, days_ahead)
                        method_used = "Simple Trend (ML evaluation failed)"
               
                # (BAGIAN BARU) Logika untuk metode Direct Prediction
                elif prediction_method == "Machine Learning (Direct)" and ML_AVAILABLE:
                    # Latih model untuk memprediksi harga 'days_ahead' ke depan secara langsung
                    model_info, feature_cols = train_direct_model(df, days_to_predict=days_ahead)
                
                    if model_info:
                    # Dapatkan satu prediksi final
                        predicted_price_final = make_direct_prediction(model_info, feature_cols, df)
                        current_price = df['harga_penutupan'].iloc[-1]

                        # Buat array prediksi dengan interpolasi linier untuk visualisasi
                        # Ini membuat garis lurus dari harga saat ini ke harga prediksi final
                        predictions = np.linspace(current_price, predicted_price_final, days_ahead)
                    
                        method_used = "Machine Learning (Direct)"
                    
                        st.info("Metode Direct Prediction melatih model untuk memprediksi titik akhir secara langsung. Metrik performa tidak ditampilkan untuk metode ini dalam implementasi saat ini.")

                    else:
                        predictions = simple_prediction(df, days_ahead)
                        method_used = "Simple Trend (ML training failed)"

                # (BAGIAN LAMA YANG DISESUAIKAN) Logika untuk metode Iterative    
                
                elif prediction_method == "Machine Learning" and ML_AVAILABLE:
                    # Train ML models
                    model_info, feature_cols, model_scores = train_ml_models(df)
                   
                    if model_info is not None:
                        predictions = make_predictions(model_info, feature_cols, df, days_ahead)
                        method_used = "Machine Learning"
                        
                        # Tampilkan performa model (hanya untuk metode iteratif)
                        if model_scores:
                            st.markdown("### 🤖 Model Performance")
                            for model_name, scores in model_scores.items():
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(f"{model_name} - MAE", f"$ {scores['MAE']:,.0f}")
                                with col2:
                                    st.metric(f"{model_name} - R²", f"{scores['R2']:.3f}")
                                with col3:
                                    st.metric(f"{model_name} - RMSE", f"$ {np.sqrt(scores['MSE']):,.0f}")
                    else:
                        predictions = simple_prediction(df, days_ahead)
                        method_used = "Simple Trend (ML training failed)"
                      
                # Logika untuk Simple Trend  
                else:
                    predictions = simple_prediction(df, days_ahead)
                    method_used = "Simple Trend"
                
                # Display prediction results
                current_price = df['harga_penutupan'].iloc[-1]
                predicted_price = predictions[-1]
                total_change = predicted_price - current_price
                change_pct = (total_change / current_price) * 100 if current_price > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'''
                    <div class="prediction-card">
                        <h3>Current Price</h3>
                        <h2>$ {current_price:,.0f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div class="prediction-card">
                        <h3>Predicted Price</h3>
                        <h2>$ {predicted_price:,.0f}</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col3:
                    color = "#28a745" if change_pct > 0 else "#dc3545"
                    arrow = "↗️" if change_pct > 0 else "↘️"
                    st.markdown(f'''
                    <div class="prediction-card">
                        <h3>Expected Change</h3>
                        <h2 style="color: {color}">{arrow} {change_pct:.1f}%</h2>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Prediction chart
                _, fig_prediction = create_charts(df, predictions, days_ahead)
                if fig_prediction:
                    st.plotly_chart(fig_prediction, use_container_width=True)
                
                # Prediction details
                st.markdown("### 📋 Prediction Details")
                
                future_dates = pd.date_range(
                    start=df['tanggal'].max() + timedelta(days=1),
                    periods=len(predictions),
                    freq='D'
                )
                
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': predictions.round(0),
                    'Daily Change': np.concatenate([[0], np.diff(predictions)]).round(0),
                    'Change %': np.concatenate([[0], np.diff(predictions) / predictions[:-1] * 100]).round(2)
                })
                
                st.dataframe(pred_df, use_container_width=True)
                
                # Download predictions
                csv = pred_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"coffee_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
                
                st.info(f"Method used: {method_used}")
    with tab_eval:
        st.markdown("### 🧪 Uji Kinerja & Diagnostik Model")
        st.info("""
            Gunakan fitur ini untuk menjalankan analisis mendalam. Proses ini akan menjalankan
            Uji Silang (Cross-Validation) untuk menemukan model terbaik, lalu menampilkan
            metrik dan grafik diagnostik lengkap untuk model tersebut.
        """)

        if st.button("🚀 Mulai Analisis Lengkap (Proses Sangat Lambat)"):
            evaluation_results_df = None
            with st.spinner("Langkah 1/2: Menjalankan Uji Silang..."):
                evaluation_results_df = evaluate_models_with_tscv(df)

            if evaluation_results_df is not None and not evaluation_results_df.empty:
                st.markdown("#### Hasil Uji Silang (Cross-Validation)")
                st.dataframe(evaluation_results_df.sort_values(by='Rata-rata MAE'), use_container_width=True)
                
                best_model_name = evaluation_results_df.loc[evaluation_results_df['Rata-rata MAE'].idxmin()]['Model']
                st.success(f"Model terbaik: **{best_model_name}**. Melanjutkan ke analisis diagnostik...")

                with st.spinner(f"Langkah 2/2: Melatih ulang {best_model_name} untuk analisis..."):
                    features_df = create_features(df)
                    feature_cols = [col for col in features_df.columns if col not in ['tanggal', 'harga_penutupan', 'perubahan_harga', 'synthesized']]
                    X = features_df[feature_cols].fillna(0)
                    y = features_df['harga_penutupan']
                    split_index = int(len(X) * 0.8)
                    X_train, X_test, y_train, y_test = X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]
                    
                    model_config = {
                        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
                        'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1),
                        'LightGBM': LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1),
                        'CatBoost': CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_state=42)
                    }
                    final_model = model_config[best_model_name]
                    
                    final_model.fit(X_train, y_train)
                    y_pred = final_model.predict(X_test)
                    
                    metrics = evaluate(y_test, y_pred)
                    fig_importance = plot_feature_importance(final_model, feature_cols)
                    fig_pred_actual = plot_prediction_vs_actual(y_test, y_pred, best_model_name)
                    fig_residuals = plot_residuals(y_test, y_pred, best_model_name)

                st.markdown(f"### Analisis Diagnostik untuk: **{best_model_name}**")
                diag_tab1, diag_tab2, diag_tab3 = st.tabs(["📊 Ringkasan Metrik", "🎯 Prediksi vs Aktual", "⚠️ Analisis Error"])
                with diag_tab1:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MAE", f"${metrics['MAE']:,.2f}")
                    col2.metric("R² Score", f"{metrics['R²']:.3f}")
                    col3.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
                    st.plotly_chart(fig_importance, use_container_width=True)
                with diag_tab2:
                    st.plotly_chart(fig_pred_actual, use_container_width=True)
                with diag_tab3:
                    st.plotly_chart(fig_residuals, use_container_width=True)
                    
    with tab3:
        st.markdown("### 📊 Data Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Price Statistics")
            price_stats = {
                "Current Price": f"$ {stats['current_price']:,.0f}",
                "Average Price": f"$ {stats['avg_price']:,.0f}",
                "Minimum Price": f"$ {stats['min_price']:,.0f}",
                "Maximum Price": f"$ {stats['max_price']:,.0f}",
                "Price Range": f"$ {stats['max_price'] - stats['min_price']:,.0f}",
                "Recent Change": f"$ {stats['price_change']:+,.0f}"
            }
            
            for key, value in price_stats.items():
                st.markdown(f'<div class="metric-card"><strong>{key}:</strong><br>{value}</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📊 Data Information")
            data_stats = {
                "Total Records": f"{stats['total_records']:,}",
                "Date Range": stats['date_range'],
                "Average Volume": f"{stats['avg_volume']:,.0f}",
                "Synthesized Data": f"{stats['synthesized_count']:,} records",
                "Real Data": f"{stats['total_records'] - stats['synthesized_count']:,} records",
                "Data Completeness": f"{((stats['total_records'] - stats['synthesized_count']) / stats['total_records'] * 100):.1f}%"
            }
            
            for key, value in data_stats.items():
                st.markdown(f'<div class="metric-card"><strong>{key}:</strong><br>{value}</div>', 
                           unsafe_allow_html=True)
        
        # Price distribution
        st.markdown("#### 📊 Price Distribution")
        fig_dist = px.histogram(df, x='harga_penutupan', nbins=50, title="Price Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab4:
        st.markdown("### 📋 Raw Data")
        
        # Data filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_rows = st.selectbox("Show rows:", [10, 25, 50, 100], index=1)
        
        with col2:
            data_type = st.selectbox("Data type:", ["All", "Real Data", "Synthesized"])
        
        with col3:
            sort_by = st.selectbox("Sort by:", ["Date (Latest)", "Date (Oldest)", "Price (High)", "Price (Low)"])
        
        # Filter and sort data
        display_df = df.copy()
        
        if data_type == "Real Data":
            display_df = display_df[display_df['synthesized'] == False]
        elif data_type == "Synthesized":
            display_df = display_df[display_df['synthesized'] == True]
        
        if sort_by == "Date (Latest)":
            display_df = display_df.sort_values('tanggal', ascending=False)
        elif sort_by == "Date (Oldest)":
            display_df = display_df.sort_values('tanggal', ascending=True)
        elif sort_by == "Price (High)":
            display_df = display_df.sort_values('harga_penutupan', ascending=False)
        elif sort_by == "Price (Low)":
            display_df = display_df.sort_values('harga_penutupan', ascending=True)
        
        # Display data
        display_columns = ['tanggal', 'harga_penutupan', 'harga_pembukaan', 'harga_tertinggi', 
                          'harga_terendah', 'volume', 'perubahan_persen', 'synthesized']
        
        st.dataframe(display_df[display_columns].head(show_rows), use_container_width=True)
        
        # Export data
        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data",
            data=csv_data,
            file_name=f"coffee_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown(f"""
        **Coffee Prediction System**
        
        **Data Source**: {stats['total_records']:,} records from {stats['date_range']}
        
        **Features**:
        - 📊 Real-time data from existing database
        - 🤖 Machine Learning predictions with multiple models
        - 📈 Interactive charts and technical analysis
        - ➕ Add new data to improve predictions
        - 📋 Comprehensive data management
        
        **Machine Learning**: {'✅ Available' if ML_AVAILABLE else '❌ Not Available (install scikit-learn)'}
        
        **Note**: This system uses your existing database with {stats['total_records']:,} records 
        and allows you to add new data to continuously improve prediction accuracy.
        """)

    st.markdown("""<hr style="margin-top: 3rem; margin-bottom: 1rem;">""", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align: center; font-size: 0.9rem; color: gray; margin-top: 2rem;'>
    © {datetime.now().year} Freddy Mazmur – Project Manager PT Sahabat Agro Group. All rights reserved.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
