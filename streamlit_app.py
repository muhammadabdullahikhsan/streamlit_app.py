import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import time
import plotly.express as px
import json
import os

# Konfigurasi tampilan Streamlit
st.set_page_config(page_title="Fraud Guard", layout="wide")

# Pilihan tema
theme = st.sidebar.selectbox("Pilih Tema:", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
        <style>
        .reportview-container {
            background: #2E2E2E;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# Kolom wajib
REQUIRED_COLUMNS = [
    'transaction_id', 'amount', 'time', 'merchant',
    'category', 'location', 'device', 'ip_address', 'user_id'
]

USER_DATA_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)

users = load_users()

def load_data(uploaded_file=None, data_url=None):
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File berhasil diunggah!")
        except Exception as e:
            st.error(f"Error saat membaca file: {e}")
    elif data_url:
        try:
            df = pd.read_csv(data_url)
            st.success("Data berhasil diambil dari URL!")
        except Exception as e:
            st.error(f"Error saat mengambil dari URL: {e}")
    else:
        # Generate sample data if no input provided
        data = {
            'transaction_id': range(1000),
            'amount': np.random.exponential(50, 1000),
            'time': np.random.uniform(0, 24, 1000),
            'merchant': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000),
            'category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'services'], 1000),
            'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'JP'], 1000),
            'device': np.random.choice(['mobile', 'desktop', 'tablet'], 1000),
            'ip_address': [f"192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}" for _ in range(1000)],
            'user_id': np.random.randint(1000, 2000, 1000)
        }
        df = pd.DataFrame(data)
        # Create some outliers
        df['amount'] = np.where(df['amount'] > 200, df['amount'] * 5, df['amount'])

    if df is not None:
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            st.error("❌ File tidak valid. Kolom wajib harus ada:\n\n" + ", ".join(REQUIRED_COLUMNS))
            return None
    return df

def convert_time_to_float(df):
    if 'time' not in df.columns:
        return df
    
    # Handle case where time might already be in float format
    if not np.issubdtype(df['time'].dtype, np.number):
        def time_to_float(t):
            try:
                h, m_ampm = t.split(':')
                m, ampm = m_ampm.split(' ')
                h = int(h)
                m = int(m)
                if ampm.upper() == 'PM' and h != 12:
                    h += 12
                if ampm.upper() == 'AM' and h == 12:
                    h = 0
                return h + m / 60.0
            except:
                return np.nan
        df['time_float'] = df['time'].apply(time_to_float)
        df = df.dropna(subset=['time_float'])
    else:
        df['time_float'] = df['time']
    
    return df

def detect_anomalies(df):
    if df.empty or len(df) < 1:
        st.error("Data tidak cukup untuk deteksi anomali (minimal 1 sampel)")
        return df
    
    df = convert_time_to_float(df)
    
    if 'amount' not in df.columns or 'time_float' not in df.columns:
        st.error("Kolom 'amount' atau 'time' tidak ditemukan dalam data")
        return df
    
    try:
        model = IsolationForest(contamination=0.05, random_state=42)
        features = df[['amount', 'time_float']].dropna()
        
        if len(features) < 1:
            st.error("Tidak ada data yang valid untuk dilatih")
            return df
            
        df['anomaly_score'] = model.fit_predict(features)
        df['anomaly'] = np.where(df['anomaly_score'] == -1, 1, 0)
    except Exception as e:
        st.error(f"Error saat mendeteksi anomali: {str(e)}")
    
    return df

def visualize_data(df):
    if df.empty:
        st.warning("Tidak ada data untuk divisualisasikan")
        return
        
    tab1, tab2, tab3 = st.tabs(["Distribusi Transaksi", "Anomali Waktu", "Peta Anomali"])
    
    with tab1:
        try:
            fig = px.histogram(df, x='amount', color='anomaly', title='Distribusi Jumlah Transaksi',
                              labels={'amount': 'Jumlah Transaksi'}, color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error membuat histogram: {str(e)}")
            
    with tab2:
        try:
            fig = px.scatter(df, x='time', y='amount', color='anomaly', title='Pola Transaksi Berdasarkan Waktu',
                            labels={'time': 'Waktu', 'amount': 'Jumlah'}, color_discrete_map={0: 'blue', 1: 'red'})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error membuat scatter plot: {str(e)}")
            
    with tab3:
        try:
            if 'anomaly' in df.columns and 'location' in df.columns:
                country_counts = df[df['anomaly'] == 1]['location'].value_counts().reset_index()
                country_counts.columns = ['location', 'fraud_count']
                fig = px.choropleth(country_counts, locations='location', locationmode='country names',
                                   color='fraud_count', title='Distribusi Fraud Berdasarkan Lokasi',
                                   color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Kolom 'anomaly' atau 'location' tidak ditemukan untuk membuat peta")
        except Exception as e:
            st.error(f"Error membuat choropleth map: {str(e)}")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

st.markdown("<h2 style='text-align: center;'>Selamat Datang di Fraud Guard</h2>", unsafe_allow_html=True)

# Login/Registration Section
col_center = st.columns(3)[1]
with col_center:
    option = st.radio("Pilih Opsi:", ("Login", "Daftar"), horizontal=True)
    
    if option == "Login":
        st.subheader("🔐 Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            if username in users and users[username]['password'] == password:
                st.session_state['logged_in'] = True
                st.session_state['current_user'] = username
                st.success("Login berhasil!")
            else:
                st.error("Username atau password salah.")
                
    else:
        st.subheader("📝 Daftar Akun")
        email_or_phone = st.text_input("Email atau No HP")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Konfirmasi Password", type="password")
        if st.button("Daftar"):
            if email_or_phone and username and password and confirm_password:
                if password == confirm_password:
                    if username not in users:
                        users[username] = {'email_or_phone': email_or_phone, 'password': password}
                        save_users(users)
                        st.success("Akun berhasil dibuat! Silakan login.")
                    else:
                        st.error("Username sudah terdaftar.")
                else:
                    st.error("Password dan konfirmasi password tidak cocok.")
            else:
                st.error("Silakan isi semua kolom.")

# Main Application after Login
if st.session_state['logged_in']:
    st.title("🛡 Sistem Deteksi Fraud Real-Time")
    st.markdown(f"Akun: {st.session_state['current_user']}")
    
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state.pop('current_user', None)
        st.experimental_rerun()

    st.markdown("Simulasi sistem deteksi transaksi mencurigakan menggunakan Isolation Forest.")

    with st.sidebar:
        st.header("📂 Input Data")
        source = st.radio("Pilih sumber data:", ("Contoh dataset", "Upload CSV", "URL dataset"))
        
        df = None
        if source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload file CSV:", type=["csv"])
            if uploaded_file is not None:
                df = load_data(uploaded_file=uploaded_file)
                
        elif source == "URL dataset":
            data_url = st.text_input("Masukkan URL:")
            if data_url:
                df = load_data(data_url=data_url)
        else:
            df = load_data()

    if df is not None and not df.empty:
        st.subheader("📊 Analisis Data Transaksi")
        
        with st.spinner("🔍 Mendeteksi anomali..."):
            df = detect_anomalies(df)
            time.sleep(1)

        total = len(df)
        fraud = df['anomaly'].sum() if 'anomaly' in df.columns else 0
        fraud_pct = (fraud / total) * 100 if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transaksi", total)
        col2.metric("Fraud Terdeteksi", fraud)
        col3.metric("Persentase Fraud", f"{fraud_pct:.2f}%")

        st.subheader("📈 Visualisasi Data")
        visualize_data(df)

        st.subheader("🚨 Transaksi Mencurigakan")
        if 'anomaly' in df.columns:
            st.dataframe(df[df['anomaly'] == 1].sort_values(by='amount', ascending=False).head(20))
        else:
            st.warning("Tidak ada data anomali yang terdeteksi")

        st.subheader("📡 Simulasi Real-Time")
        if st.button("Tambah Transaksi Baru"):
            new_data = {
                'transaction_id': max(df['transaction_id']) + 1,
                'amount': np.random.exponential(50) * (10 if np.random.random() > 0.95 else 1),
                'time': np.random.uniform(0, 24),
                'merchant': np.random.choice(['A', 'B', 'C', 'D', 'E']),
                'category': np.random.choice(['retail', 'food', 'travel', 'entertainment', 'services']),
                'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'JP']),
                'device': np.random.choice(['mobile', 'desktop', 'tablet']),
                'ip_address': f"192.168.{np.random.randint(0,255)}.{np.random.randint(0,255)}",
                'user_id': np.random.randint(1000, 2000)
            }
            new_df = pd.DataFrame([new_data])
            new_df = detect_anomalies(new_df)

            if new_df is not None and 'anomaly' in new_df.columns:
                if new_df['anomaly'].iloc[0] == 1:
                    st.warning("🚨 TRANSAKSI MENCURIGAKAN TERDETEKSI!")
                    st.write(new_df)
                else:
                    st.success("✅ Transaksi normal.")
                    st.write(new_df)
                    
                df = pd.concat([df, new_df], ignore_index=True)
                st.experimental_rerun()
              

st.markdown("---")
st.markdown("Kolom Wajib: " + ", ".join(REQUIRED_COLUMNS))
