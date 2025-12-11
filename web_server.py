# server.py

from fastapi import FastAPI, File, UploadFile
from datetime import datetime
import os
import json
import threading
import asyncio
from predict_picture import predict_image
from predict_voice import predict_audio
from PIL import Image
import soundfile as sf

app = FastAPI()

RESULTS_FILE = "results.json"

def init_results_file():
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            json.dump({"face": [], "voice": []}, f)

init_results_file()

def save_result(kind, data):
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    results[kind].append(data)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f)

@app.post("/picture")
async def receive_picture(file: UploadFile = File(...)):
    # Simpan file sementara
    filepath = f"temp_{file.filename}"
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Proses dengan model ML
        image = Image.open(filepath)
        hasil_prediksi, akurasi = predict_image(image)
        akurasi_percent = akurasi * 100

        # Simpan hasil
        result_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hasil Prediksi": hasil_prediksi,
            "Akurasi (%)": akurasi_percent,
            "Status": "Selesai",
            "Keterangan": f"Diproses dari {file.filename}"
        }
        save_result("face", result_data)

        os.remove(filepath)
        return {"status": "success", "result": hasil_prediksi}
    except Exception as e:
        os.remove(filepath)
        return {"status": "error", "message": str(e)}

@app.post("/voice")
async def receive_voice(file: UploadFile = File(...)):
    # Simpan file sementara
    filepath = f"temp_{file.filename}"
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Proses dengan model ML
        hasil_prediksi, akurasi = predict_audio(filepath)
        akurasi_percent = akurasi * 100

        # Simpan hasil
        result_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Hasil Prediksi": hasil_prediksi,
            "Akurasi (%)": akurasi_percent,
            "Status": "Selesai",
            "Keterangan": f"Diproses dari {file.filename}"
        }
        save_result("voice", result_data)

        os.remove(filepath)
        return {"status": "success", "result": hasil_prediksi}
    except Exception as e:
        os.remove(filepath)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)