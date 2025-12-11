import pickle
import cv2
import gdown
import numpy as np
from PIL import Image

IMG_SIZE = 96
class_names = ['ANGGI_FACES', 'DEVI_FACES', 'FARIDA_FACES', 'ILHAM_FACES', 'OTHER_FACES']
file_id = "1OsMc-fey6Z2vwuZ815QwI7JVtinynOIJ"
Path_gdrive = f"gdown {file_id}"
# Load model dan scaler
try:
    with open('image_svc_model.pkl', 'rb') as file:
        svc_model = pickle.load(Path)
    with open('image_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Model files not found. Please ensure 'image_svc_model.pkl' and 'image_scaler.pkl' are in the same directory.")

def preprocess_image(image, img_size=IMG_SIZE, scaler=scaler):
    img_np = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_np, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_flattened = img_normalized.reshape(1, -1)
    img_scaled = scaler.transform(img_flattened)
    return img_scaled

def predict_image(image):
    """
    Fungsi untuk memprediksi gambar.
    Input: PIL Image
    Output: predicted_class_name (str), confidence (float)
    """
    processed_image_data = preprocess_image(image)
    prediction_index = svc_model.predict(processed_image_data)[0]
    prediction_proba = svc_model.predict_proba(processed_image_data)[0]

    predicted_class_name = class_names[prediction_index]
    confidence = np.max(prediction_proba)

    return predicted_class_name, confidence