import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, accuracy_score
import time
import plotly.express as px

# Konfigurasi tampilan Streamlit
st.set_page_config(page_title="Fraud Guard", layout="wide")

# Kolom wajib
REQUIRED_COLUMNS = [
    'transaction_id', 'amount', 'time'
]

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
        }
        df = pd.DataFrame(data)
        # Create some outliers
        df['amount'] = np.where(df['amount'] > 200, df['amount'] * 5, df['amount'])

    if df is not None:
        if not set(REQUIRED_COLUMNS).issubset(df.columns):
            st.error("❌ File tidak valid. Kolom wajib harus ada:\n\n" + ", ".join(REQUIRED_COLUMNS))
            return None
    return df

def detect_anomalies(df):
    if df.empty or len(df) < 1:
        st.error("Data tidak cukup untuk deteksi anomali (minimal 1 sampel)")
        return df
    
    try:
        model = IsolationForest(contamination=0.05, random_state=42)
        features = df[['amount', 'time']].dropna()
        
        if len(features) < 1:
            st.error("Tidak ada data yang valid untuk dilatih")
            return df
            
        df['anomaly_score'] = model.fit_predict(features)
        df['anomaly'] = np.where(df['anomaly_score'] == -1, 1, 0)
    except Exception as e:
        st.error(f"Error saat mendeteksi anomali: {str(e)}")
    
    return df

def evaluate_anomaly_detection(df):
    if 'anomaly' not in df.columns:
        st.warning("Kolom 'anomaly' tidak ditemukan dalam data")
        return None, None, None

    y_true = np.zeros(len(df))  # Assume all normal transactions
    y_true[df['anomaly'] == 1] = 1  # Set actual anomalies to 1

    precision = precision_score(y_true, df['anomaly'])
    recall = recall_score(y_true, df['anomaly'])
    accuracy = accuracy_score(y_true, df['anomaly'])

    return precision, recall, accuracy

def visualize_data(df):
    if df.empty:
        st.warning("Tidak ada data untuk divisualisasikan")
        return
        
    try:
        fig = px.histogram(df, x='amount', color='anomaly', title='Distribusi Jumlah Transaksi',
                            labels={'amount': 'Jumlah Transaksi'}, color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error membuat histogram: {str(e)}")
    
    try:
        fig = px.scatter(df, x='time', y='amount', color='anomaly', title='Pola Transaksi Berdasarkan Waktu',
                        labels={'time': 'Waktu', 'amount': 'Jumlah'}, color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error membuat scatter plot: {str(e)}")

# Main Application
st.title("🛡 Sistem Deteksi Fraud Real-Time")

# Load Data Section
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

    precision, recall, accuracy = evaluate_anomaly_detection(df)

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

    st.subheader("📊 Evaluasi Deteksi Anomali")
    st.write(f"Presisi: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"Akurasi: {accuracy:.2f}")

    st.subheader("📡 Simulasi Real-Time")
    if st.button("Tambah Transaksi Baru"):
        new_data = {
            'transaction_id': max(df['transaction_id']) + 1,
            'amount': np.random.exponential(50) * (10 if np.random.random() > 0.95 else 1),
            'time': np.random.uniform(0, 24),
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
