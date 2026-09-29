import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

from langchain_core.tools import tool
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent

cifar_model = None

# cifar_model = keras.models.load_model("cifar-10.keras")
CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]
@tool
def predict_cifar10(file_path: str) -> dict:
    """Use this tool to classify an image using the CIFAR-10 model. Input must be a valid file path."""

    global cifar_model

    if cifar_model is None:
        # TensorFlow is deliberately lazy-loaded.  Importing it at module load
        # time changes CUDA's process state and can cause CPU YOLO inference
        # (PyTorch) to stall before its first prediction.
        from tensorflow import keras

        cifar_model = keras.models.load_model(BASE_DIR / "cifar-10.keras")

    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((32, 32))
        
        img_array = np.array(img) / 255.0
        
        
        input_data = np.expand_dims(img_array, axis=0)
        
        # 4. Run inference
        predictions = cifar_model.predict(input_data)
        
        # 5. Extract the winning class and its confidence score
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return {
            "status": "success",
            "predicted_class": CIFAR10_CLASSES[predicted_idx],
            "confidence": round(confidence, 4)
        }
    except Exception as e:
         return {"status": "error", "message": str(e)}
