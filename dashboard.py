import streamlit as st
import paho.mqtt.client as mqtt
import pandas as pd
import time
from datetime import datetime
import json
import threading
import plotly.graph_objects as go
import numpy as np
import os # Wajib ditambahkan untuk client ID unik

# PENTING: Mengaktifkan Wide Layout
st.set_page_config(layout="wide")

# ====================================================================
# KONFIGURASI TOPIK MQTT
# ====================================================================

mqtt_broker = "broker.hivemq.com"
mqtt_port = 1883
topic_brankas = "data/status/kontrol"
topic_face = "ai/face/result"
topic_voice = "ai/voice/result"
topic_cam_url = "/iot/camera/photo"
topic_dist = "data/dist/kontrol"
topic_pir = "data/pir/kontrol"
topic_alarm_control = "data/alarm/kontrol"
topic_cam_trigger = "/iot/camera/trigger"
topic_audio_link = "data/audio/link"

# ====================================================================
# INISIALISASI SESSION STATE (Hanya MQTT Data)
# ====================================================================

if 'mqtt_queue' not in st.session_state:
    st.session_state.mqtt_queue = [] 

if 'data_brankas' not in st.session_state:
    st.session_state.data_brankas = pd.DataFrame(columns=[
        "Timestamp", "Status Brankas", "Jarak (cm)", "PIR", "Prediksi Wajah", "Prediksi Suara", "Label Prediksi"
    ])
if 'data_face' not in st.session_state:
    st.session_state.data_face = pd.DataFrame(columns=["Timestamp", "Hasil Prediksi", "Status", "Keterangan"])
if 'data_voice' not in st.session_state:
    st.session_state.data_voice = pd.DataFrame(columns=["Timestamp", "Hasil Prediksi", "Status", "Keterangan"])

if 'photo_url' not in st.session_state:
    st.session_state.photo_url = "https://via.placeholder.com/640x480?text=Menunggu+Foto"
if 'audio_url' not in st.session_state:
    st.session_state.audio_url = None

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
    
# ====================================================================
# FUNGSI LOGIKA PREDIKSI GABUNGAN
# ====================================================================

def generate_final_prediction(row):
    wajah = row.get("Prediksi Wajah", "")
    suara = row.get("Prediksi Suara", "")
    jarak = row.get("Jarak (cm)", np.nan)
    pir = row.get("PIR", np.nan)
    status = row.get("Status Brankas", "")

    if "Brangkas Dibuka Paksa" in status:
        return "⚠ Dibobol!"
    if wajah in ["Unknown", "OTHER_FACES"] or suara == "Not_User": 
        return "🚨 Mencurigakan!"
    if "Terbuka Secara Aman" in status:
        return "✅ Sah & Aman"
    # Diperbaiki: Cek jika jarak adalah float yang valid
    if pd.notna(jarak) and isinstance(jarak, (int, float)) and jarak > 0 and jarak < 25: 
        return "pintu brangkastertutup"
    if pd.notna(pir) and pir == 1:
        return "👀 Gerakan Terdeteksi"
    return "✅ Aman"

# ====================================================================
# FUNGSI CALLBACK MQTT & PROSES ANTRIAN
# ====================================================================

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe([
            (topic_brankas, 0), (topic_face, 0), (topic_voice, 0), 
            (topic_cam_url, 0), (topic_dist, 0), (topic_pir, 0), 
            (topic_audio_link, 0)
        ])
        st.info("✅ Klien MQTT Terkoneksi & Berhasil Berlangganan Topik.")
    else:
        st.error(f"❌ Koneksi MQTT Gagal, kode: {rc}")


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8").strip()
        # Thread hanya menambahkan pesan ke antrian (aman)
        st.session_state.mqtt_queue.append({
            "topic": msg.topic,
            "payload": payload,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except (UnicodeDecodeError, AttributeError):
        pass

# ====================================================================
# INISIALISASI KLIEN MQTT DENGAN CACHE (SOLUSI UNTUK OSERROR: TOO MANY OPEN FILES)
# ====================================================================

@st.cache_resource # <--- DEKORATOR WAJIB!
def get_mqtt_client(broker, port):
    """Membuat, menghubungkan, dan memulai loop klien MQTT hanya sekali."""
    client_id = f"StreamlitDashboard-{os.getpid()}-{int(time.time())}" # ID unik
    
    try:
        client = mqtt.Client(client_id=client_id, clean_session=True)
        client.on_connect = on_connect
        client.on_message = on_message
        
        # Coba koneksi
        client.connect(broker, port, 60)
        client.loop_start() 
        return client

    except Exception as e:
        # Jika koneksi gagal, kembalikan None
        st.error(f"FATAL: Gagal koneksi MQTT saat inisialisasi: {e}")
        return None 

# ====================================================================
# PROSES ANTRIAN DI THREAD UTAMA STREAMLIT
# ====================================================================
def process_mqtt_queue():
    if not st.session_state.mqtt_queue:
        return
        
    messages_to_process = list(st.session_state.mqtt_queue)
    st.session_state.mqtt_queue = [] # Kosongkan antrian

    should_rerun = False
    
    for msg in messages_to_process:
        topic = msg['topic']
        payload = msg['payload']
        timestamp = msg['time']
        
        # --- LOGIKA UNTUK data_brankas (Sensor) ---
        if topic == topic_brankas:
            new_row = {
                "Timestamp": timestamp,
                "Status Brankas": payload,
                "Jarak (cm)": np.nan,
                "PIR": np.nan,
                "Prediksi Wajah": "Menunggu...",
                "Prediksi Suara": "Menunggu...",
                "Label Prediksi": "Belum Diproses"
            }
            # Tambahkan baris baru
            st.session_state.data_brankas = pd.concat([
                st.session_state.data_brankas, pd.DataFrame([new_row])
            ], ignore_index=True)
            should_rerun = True # Data baru, harus refresh
            
        elif not st.session_state.data_brankas.empty:
            last_index = st.session_state.data_brankas.index[-1]
            
            # Update data sensor di baris terakhir
            if topic == topic_dist:
                try:
                    st.session_state.data_brankas.loc[last_index, 'Jarak (cm)'] = float(payload)
                    should_rerun = True
                except ValueError: pass
            elif topic == topic_pir:
                try:
                    st.session_state.data_brankas.loc[last_index, 'PIR'] = int(payload)
                    should_rerun = True
                except ValueError: pass
            
            # --- LOGIKA UNTUK ML HASIL & URL ---
            elif topic == topic_face:
                st.session_state.data_brankas.loc[last_index, 'Prediksi Wajah'] = payload
                st.session_state.data_face = pd.concat([st.session_state.data_face, 
                                                         pd.DataFrame([{"Timestamp": timestamp, "Hasil Prediksi": payload, "Status": "Success", "Keterangan": "MQTT Live Result"}])], ignore_index=True)
                should_rerun = True
            
            elif topic == topic_voice:
                st.session_state.data_brankas.loc[last_index, 'Prediksi Suara'] = payload
                st.session_state.data_voice = pd.concat([st.session_state.data_voice, 
                                                          pd.DataFrame([{"Timestamp": timestamp, "Hasil Prediksi": payload, "Status": "Success", "Keterangan": "MQTT Live Result"}])], ignore_index=True)
                should_rerun = True
            
            elif topic == topic_cam_url:
                st.session_state.photo_url = f"{payload}?t={int(time.time())}"
                should_rerun = True
            
            elif topic == topic_audio_link:
                st.session_state.audio_url = f"{payload}?t={int(time.time())}" 
                should_rerun = True

    # Update label prediksi akhir (Dilakukan setelah semua update loc)
    if not st.session_state.data_brankas.empty:
        st.session_state.data_brankas["Label Prediksi"] = st.session_state.data_brankas.apply(
            generate_final_prediction, axis=1
        )
        
    return should_rerun # Mengembalikan apakah Streamlit harus di-rerun

# ====================================================================
# TAMPILAN DASHBOARD UTAMA
# ====================================================================

# 1. Panggil klien MQTT yang di-cache (Hanya sekali)
mqtt_client = get_mqtt_client(mqtt_broker, mqtt_port)

if not mqtt_client:
    st.error("Dashboard tidak dapat terhubung ke MQTT Broker. Mohon periksa konfigurasi dan pastikan broker online.")
    st.stop() # Hentikan eksekusi jika gagal

st.title("🔒 Dashboard Monitoring Brankas & AI")

# --- BAGIAN ATAS (CHART & FOTO) ---
chart_col, photo_col = st.columns([2, 1])

with chart_col:
    st.header("Live Chart (Jarak & PIR)")
    df_plot = st.session_state.data_brankas.tail(200).copy()
    df_plot.dropna(subset=['Jarak (cm)', 'PIR'], inplace=True) 

    if not df_plot.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Timestamp"], y=df_plot["Jarak (cm)"], mode="lines+markers", name="Jarak (cm)"))
        fig.add_trace(go.Scatter(x=df_plot["Timestamp"], y=df_plot["PIR"], mode="lines+markers", name="PIR (Gerakan)", yaxis="y2"))

        fig.update_layout(
            yaxis=dict(title="Jarak (cm)"),
            yaxis2=dict(title="PIR (0=Aman, 1=Gerak)", overlaying="y", side="right", showgrid=False, range=[-0.1, 1.1]),
            height=520,
            legend=dict(x=0, y=1.1, orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Menunggu data Jarak/PIR untuk menampilkan chart.")
    
    csv_data = st.session_state.data_brankas.to_csv(index=False).encode("utf-8")
    is_data_available = not st.session_state.data_brankas.empty
    
    if is_data_available:
        st.download_button(
            label="⬇️ Download Semua Log CSV",
            data=csv_data,
            file_name=f"brankas_logs_{int(time.time())}.csv",
            mime="text/csv"
        )
    else:
        st.info("Tidak ada data log untuk diunduh.")


with photo_col:
    st.header("Foto Terbaru")
    st.image(st.session_state.photo_url, use_column_width=True)
    
    st.markdown("### Kontrol Cepat")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        if st.button("📷 Ambil Foto"): 
            mqtt_client.publish(topic_cam_trigger, "capture")
            
    with control_col2:
        if st.button("🔇 Matikan Alarm"):
            mqtt_client.publish(topic_alarm_control, "OFF")
            
    with control_col3:
        if st.button("🔄 Refresh Foto"):
            mqtt_client.publish(topic_cam_trigger, "capture")
            
st.markdown("---")

# --- BAGIAN BAWAH (TAB UNTUK DETAIL LOG) ---
tab1, tab2, tab3 = st.tabs(["🏠 Detail Brankas", "🖼️ Log Prediksi Wajah", "🔊 Log Prediksi Suara"])

with tab1:
    st.subheader("Log Data Brankas Lengkap")
    st.dataframe(st.session_state.data_brankas, use_container_width=True)

with tab2:
    st.subheader("Log Prediksi Wajah (Keputusan dari Web Server)")
    st.dataframe(st.session_state.data_face.tail(10), use_container_width=True)

with tab3:
    st.subheader("Audio Terbaru untuk Analisis")
    if st.session_state.audio_url:
        st.audio(st.session_state.audio_url, format='audio/wav')
    else:
        st.info("Menunggu rekaman audio terbaru...")

    st.subheader("Log Prediksi Suara (Keputusan dari Web Server)")
    st.dataframe(st.session_state.data_voice.tail(10), use_container_width=True)


# ====================================================================
# PENTING: PENGATURAN REFRESH OTOMATIS BERDASARKAN DATA BARU
# ====================================================================

# 1. Proses semua pesan di antrian
should_rerun = process_mqtt_queue()

# 2. Cek apakah ada data baru atau waktu refresh berkala telah tiba
# Refresh jika ada data MQTT baru ATAU jika 5 detik telah berlalu (untuk update status umum)
if should_rerun or (time.time() - st.session_state.last_refresh > 5): 
    st.session_state.last_refresh = time.time()
    st.rerun()
