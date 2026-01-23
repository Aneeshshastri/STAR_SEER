import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from CONFIGS import Config


print("⏳ Loading Model & Stats...")

# Load Stats
try:
    with np.load("model/dataset_stats.npz") as data:
        # Ensure float32 for TensorFlow compatibility
        MEANS_ARRAY = data['mean'].astype(np.float32)
        STDS_ARRAY = data['std'].astype(np.float32)
        
    # Create a lookup dictionary for "Auto-Fill" functionality
    # Maps 'TEFF' -> 4500.0, 'LOGG' -> 2.5, etc.
    DEFAULT_MEANS = {label: float(MEANS_ARRAY[i]) for i, label in enumerate(Config.SELECTED_LABELS)}
    
    print(f"✅ Stats loaded. Found {len(MEANS_ARRAY)} features.")
    
    # Sanity Check
    if len(MEANS_ARRAY) != len(Config.SELECTED_LABELS):
        print(f"⚠️ WARNING: Stats file has {len(MEANS_ARRAY)} values, but Config.SELECTED_LABELS has {len(Config.SELECTED_LABELS)}.")

except Exception as e:
    print(f"❌ Error loading stats: {e}")
    # Fallback for testing ONLY (prevents crash if file is missing)
    MEANS_ARRAY = np.zeros(len(Config.SELECTED_LABELS), dtype=np.float32)
    STDS_ARRAY = np.ones(len(Config.SELECTED_LABELS), dtype=np.float32)
    DEFAULT_MEANS = {label: 0.0 for label in Config.SELECTED_LABELS}


# Load Model
try:
    model = tf.keras.models.load_model("model/stellar_generator.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    model = None

# --- 3. FastAPI Setup ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input Model: All fields optional (defaulting to None)
class StellarParams(BaseModel):
    params: Dict[str, Optional[float]]

def generate_wavelengths():
    """Generates the x-axis for the spectrum."""
    return np.logspace(np.log10(Config.WAVELENGTH_START), np.log10(Config.WAVELENGTH_END), Config.OUTPUT_LENGTH)

@app.post("/predict")
async def predict_spectrum(user_input: StellarParams):
    """
    1. Receives partial or full user input.
    2. Fills missing keys with the dataset mean.
    3. Normalizes the vector.
    4. Returns model prediction.
    """
    user_dict = user_input.params
    
    # --- Step A: Vector Construction (Handle Missing Inputs) ---
    input_vector = []
    
    for i, label in enumerate(Config.SELECTED_LABELS):
        # Check if user provided this specific label
        val = user_dict.get(label)
        
        # If input is None or empty string, use the DEFAULT MEAN from .npz
        if val is None or val == "":
            val = DEFAULT_MEANS[label]
        
        input_vector.append(float(val))

    # --- Step B: Normalization ---
    # Formula: (x - mean) / std
    raw_array = np.array(input_vector, dtype=np.float32)
    normalized_array = (raw_array - MEANS_ARRAY) / (STDS_ARRAY + 1e-7)
    
    # Reshape for model (1, n_features)
    model_input = normalized_array.reshape(1, -1)

    # --- Step C: Inference ---
    if model:
        # Predict returns (1, Config.OUTPUT_LENGTH)
        prediction = model.predict(model_input, verbose=0)
        flux_output = prediction[0].tolist()
    else:
        # Mock output for testing UI without model
        x = np.linspace(0, 50, Config.OUTPUT_LENGTH)
        flux_output = (np.sin(x) + raw_array[0]/5000).tolist() # Dummy math

    return {
        "wavelengths": generate_wavelengths().tolist(),
        "flux": flux_output,
        "used_values": dict(zip(Config.SELECTED_LABELS, input_vector)) # Send back what values were actually used
    }

# Mount static files (frontend)
app.mount("/", StaticFiles(directory="C:/Users/Aneesh Shastri/OneDrive/Documents/GitHub/STAR_SEER/Model_inference/static", html=True), name="static")