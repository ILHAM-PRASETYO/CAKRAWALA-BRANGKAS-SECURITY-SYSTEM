# server.py (Setelah Direvisi)

from fastapi import FastAPI
from datetime import datetime
import os
import json
import requests # <--- DITAMBAHKAN
from predict_picture import predict_image
from predict_voice import predict_audio
from PIL import Image
import soundfile as sf
import paho.mqtt.client as mqtt # <--- DITAMBAHKAN untuk komunikasi ke Streamlit

app = FastAPI()

# --- MQTT SETUP ---
MQTT_SERVER = "broker.hivemq.com"
MQTT_PORT = 1883
TOPIC_ML_FACE_RESULT = "ai/face/result"
TOPIC_ML_VOICE_RESULT = "ai/voice/result"
TOPIC_CAM_PHOTO_URL = "/iot/camera/photo" # Untuk URL foto di Streamlit
TOPIC_AUDIO_LINK = "data/audio/link"       # Untuk URL audio di Streamlit

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_SERVER, MQTT_PORT, 60)
mqtt_client.loop_start() 
# --- END MQTT SETUP ---

# Hapus semua logika results.json (init_results_file dan save_result) 
# karena kita akan menggunakan MQTT 100% untuk status real-time.

# =================================================================
# ENDPOINT BARU: Menerima URL dan Melakukan HTTP GET (PULL)
# =================================================================

@app.get("/process")
async def process_media_from_url(url: str, media_type: str):
    """
    Endpoint ini menerima URL (alamat file di ESP32) dan tipe media.
    Kemudian, ia melakukan HTTP GET untuk mengambil file tersebut, memprosesnya, 
    dan mengirim hasilnya ke Streamlit via MQTT.
    """
    
    if not url.startswith("http"):
        return {"status": "error", "message": "URL tidak valid."}
        
    filepath = None
    
    try:
        # 1. LAKUKAN HTTP GET KE URL ESP32
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise exception jika 4xx atau 5xx error
        
        # Tentukan ekstensi dan path
        ext = ".jpg" if media_type == "picture" else ".wav"
        filepath = f"temp_media_{datetime.now().timestamp()}{ext}"
        
        # Simpan file yang didownload
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        hasil_prediksi = "N/A"
        
        # 2. PROSES DENGAN MODEL ML
        if media_type == "picture":
            image = Image.open(filepath)
            hasil_prediksi, akurasi = predict_image(image)
            # Kirim URL foto ke Streamlit untuk ditampilkan (jika perlu)
            mqtt_client.publish(TOPIC_CAM_PHOTO_URL, url) 
            # Kirim hasil ML
            mqtt_client.publish(TOPIC_ML_FACE_RESULT, hasil_prediksi)
            
        elif media_type == "voice":
            hasil_prediksi, akurasi = predict_audio(filepath)
            # Kirim URL audio ke Streamlit untuk ditampilkan (jika perlu)
            mqtt_client.publish(TOPIC_AUDIO_LINK, url)
            # Kirim hasil ML
            mqtt_client.publish(TOPIC_ML_VOICE_RESULT, hasil_prediksi)

        # 3. KIRIM RESPON KE YANG MENGIRIM PERINTAH (ESP32)
        os.remove(filepath)
        return {"status": "success", "result": hasil_prediksi, "topic_sent": TOPIC_ML_FACE_RESULT if media_type == "picture" else TOPIC_ML_VOICE_RESULT}
        
    except requests.exceptions.RequestException as req_e:
        # Jika gagal mengambil file dari ESP32
        if filepath and os.path.exists(filepath): os.remove(filepath)
        return {"status": "error", "message": f"Gagal mengambil file dari URL: {str(req_e)}"}
        
    except Exception as e:
        # Jika gagal di proses ML
        if filepath and os.path.exists(filepath): os.remove(filepath)
        return {"status": "error", "message": f"Gagal proses ML: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
