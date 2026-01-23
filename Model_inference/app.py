import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from CONFIGS import Config
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras import layers, models, Input, callbacks, regularizers

@register_keras_serializable()
def scaled_sigmoid(x):
    return 1.3 * tf.nn.sigmoid(x)-0.15

@register_keras_serializable()
def sobolev_loss(y_true, y_pred):
    real_flux = y_true[:, :, 0:1]
    ivar = y_true[:, :, 1:2]
    valid_mask = tf.cast(real_flux > Config.BADPIX_CUTOFF, tf.float32)    
    safe_flux = tf.where(valid_mask == 1.0, real_flux, y_pred)
    ivar_safe = tf.clip_by_value(ivar / 1000.0, 0.0, 1.0)# scale and clip
    weight=tf.where(((safe_flux<0.9) & (ivar>0)),tf.maximum(ivar_safe,tf.cast(1.0,dtype=tf.float32)),ivar_safe)
    #chi2
    wmse_term = tf.square(safe_flux - y_pred) * weight * valid_mask
    # calculate "gradients" (difference between adjacent pixels)
    true_grad = safe_flux[:, 1:, :] - safe_flux[:, :-1, :]
    pred_grad = y_pred[:, 1:, :] - y_pred[:, :-1, :]
    
    # Calculate Squared Error of gradients (sobolev loss term)
    grad_sq_diff = tf.square(true_grad - pred_grad)
    grad_mask = valid_mask[:, 1:, :] * valid_mask[:, :-1, :]
    grad_trust = (weight[:, 1:, :] * weight[:, :-1, :])
    # Apply mask to gradient loss
    grad_loss = grad_sq_diff * grad_mask * grad_trust
    
    #pad last pixel
    grad_loss = tf.pad(grad_loss, [[0,0], [0,1], [0,0]])

@tf.keras.saving.register_keras_serializable(package="Custom")
def spectral_focus_loss(y_true, y_pred):
    """
    A loss function designed specifically for the Refiner Stage.
    It aggressively ignores continuum noise and focuses on absorption lines.
    """
    real_flux = y_true[:, :, 0:1]
    ivar = y_true[:, :, 1:2]
    
    # 1. Mask Poison Data (-9999)
    valid_mask = tf.cast(real_flux >  Config.BADPIX_CUTOFF, tf.float32)
    safe_flux = tf.where(valid_mask == 1.0, real_flux, y_pred)
    
    # 2. Scale Ivar
    ivar_safe = tf.clip_by_value(ivar / 1000.0, 0.0, 1.0)
    weight=tf.where(((safe_flux<0.9) & (ivar>0)),tf.maximum(ivar_safe,tf.cast(1.0,dtype=tf.float32)),ivar_safe)
    # 3. DEFINE ZONES
    # Line Zone: Where physics happens (Flux < 0.9)
    # Continuum Zone: Where noise happens (Flux >= 0.9)
    is_line = tf.cast(safe_flux < 0.9, tf.float32)
    is_continuum = 1.0 - is_line
    
    # 4. ASSIGN WEIGHTS
    # Lines: Weight 15 (High Priority)
    # Continuum: Weight 1 (Low Priority - prevent drift, but ignore noise)
    zone_weight = (is_line *15.0) + (is_continuum * 1.0)
    
    # 5. Combined Weight
    # We multiply by ivar to ensure we don't fit dead pixels in the line region
    final_weight = zone_weight * weight * valid_mask
    
    # 6. Calculate MSE
    squared_diff = tf.square(safe_flux - y_pred)
    weighted_loss = squared_diff * final_weight
    
    # 7. Safety
    loss = tf.where(tf.math.is_finite(weighted_loss), weighted_loss, tf.zeros_like(weighted_loss))
    
    return tf.reduce_mean(loss)

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
    model = tf.keras.models.load_model("model/final_model.keras")
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
    raw_array = raw_array.reshape(1, -1)
    teff_idx = Config.SELECTED_LABELS.index('TEFF')
    teff_vals = raw_array[:, teff_idx]     
    inv_teff = 5040.0 / (teff_vals + 1e-6)
    inv_teff = inv_teff.reshape(-1, 1)
    raw_array = np.hstack([raw_array, inv_teff])
    normalized_array = (raw_array - MEANS_ARRAY) / (STDS_ARRAY + 1e-7)
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