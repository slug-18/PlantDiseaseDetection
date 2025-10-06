from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import numpy as np
from PIL import Image
import io
import uvicorn

# === FastAPI App ===
app = FastAPI(
    title="🌿 Plant Disease Detection API",
    version="1.1",
    description="Detects plant leaf diseases using a fine-tuned ResNet50 model"
)

# === Enable CORS for frontend ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later (e.g. ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load Model ===
MODEL_PATH = "D:/PlantDiseaseDetection/models/plant_disease_resnet50.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)

# Replace these with your actual trained class labels
CLASS_NAMES = ['Pepper__bell___Bacterial_spot', 
               'Pepper__bell___healthy', 
               'Potato___Early_blight', 
               'Potato___Late_blight', 
               'Potato___healthy', 
               'Tomato_Bacterial_spot', 
               'Tomato_Early_blight', 
               'Tomato_Late_blight', 
               'Tomato_Leaf_Mold', 
               'Tomato_Septoria_leaf_spot', 
               'Tomato_Spider_mites_Two_spotted_spider_mite', 
               'Tomato__Target_Spot', 
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 
               'Tomato__Tomato_mosaic_virus', 
               'Tomato_healthy'
               ]

# === Routes ===

@app.get("/")
def home():
    """Health check route."""
    return {"message": "🌱 Plant Disease Detection API is live!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the disease class from a leaf image."""
    try:
        contents = await file.read()

        # Load image and preprocess
        img = Image.open(io.BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # ✅ same as training

        # Predict
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class {class_idx}"

        return {
            "class_index": class_idx,
            "class_name": class_name,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:
        return {"error": str(e)}


# === Run app (for local development) ===
if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
