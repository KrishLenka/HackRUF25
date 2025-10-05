# app.py
import io
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(
    app,
    resources={r"/*": {"origins": ["http://localhost:5173", "http://127.0.0.1:5173"]}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# TensorFlow / Keras
import tensorflow as tf

# PyTorch
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import timm  # if your PyTorch model uses timm

# ONNX Runtime
import onnxruntime as ort

# ---------------------------
# CONFIG - update these paths
# ---------------------------
MODEL_DIR = Path("models")
TF_MODEL_PATH = MODEL_DIR / "tf_skin_model"    # either SavedModel dir or .h5
PYTORCH_MODEL_PATH = MODEL_DIR / "pt_skin_model.pth"
ONNX_MODEL_PATH = MODEL_DIR / "skin_model.onnx"

# Load class names from training (will be auto-populated during training)
# Default classes for the 10-class skin condition dataset
CLASS_NAMES = [
    "Actinic keratosis",
    "Atopic Dermatitis",
    "Benign keratosis",
    "Dermatofibroma",
    "Healthy Skin",
    "Melanocytic nevus",
    "Melanoma",
    "Squamous cell carcinoma",
    "Tinea Ringworm Candidiasis",
    "Vascular lesion"
]
IMG_SIZE = 224
# Automatically detect best available device: NVIDIA CUDA > Apple Silicon MPS > CPU
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# ---------------------------

app = Flask(__name__)

# ---------------------------
# Preprocessing helpers
# ---------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def pil_to_numpy(img: Image.Image, size=IMG_SIZE) -> np.ndarray:
    """Resize and return HxWxC float32 image scaled to [0,1]."""
    img = img.convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr

def normalize_imagenet(img_np: np.ndarray) -> np.ndarray:
    """Normalize using ImageNet mean/std. Expects HWC float in [0,1]."""
    return (img_np - IMAGENET_MEAN) / IMAGENET_STD

def prepare_for_tf(img_np: np.ndarray) -> np.ndarray:
    """Return batched NHWC float32 for TF."""
    x = normalize_imagenet(img_np)
    x = np.expand_dims(x.astype(np.float32), axis=0)
    return x

def prepare_for_torch(img_np: np.ndarray) -> torch.Tensor:
    """Return batched NCHW torch tensor on DEVICE."""
    x = normalize_imagenet(img_np)
    x = np.transpose(x, (2,0,1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)
    t = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
    return t

def prepare_for_onnx(img_np: np.ndarray) -> np.ndarray:
    """Return batched input for ONNX (NCHW) as float32 numpy."""
    x = normalize_imagenet(img_np)
    x = np.transpose(x, (2,0,1))  # CHW
    x = np.expand_dims(x, axis=0)
    return x.astype(np.float32)

# ---------------------------
# Model loaders
# ---------------------------
print("="*70)
print("INITIALIZING SKIN CONDITION CLASSIFIER API")
print("="*70)
print(f"Device: {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
elif DEVICE == "mps":
    print(f"GPU: Apple Silicon (MPS)")
else:
    print(f"GPU: Not available - using CPU")
print("="*70)
print("\nLoading models...")

# 1) TensorFlow / Keras
tf_model = None
try:
    if TF_MODEL_PATH.suffix == ".h5":
        tf_model = tf.keras.models.load_model(str(TF_MODEL_PATH))
    else:
        tf_model = tf.keras.models.load_model(str(TF_MODEL_PATH))  # works for SavedModel dir also
    print("Loaded TensorFlow model.")
except Exception as e:
    print("Warning: TF model load failed:", e)
    tf_model = None

# 2) PyTorch model
pt_model = None
try:
    # Define the same architecture you trained. Example using timm EfficientNet_b0:
    class PTWrapper(torch.nn.Module):
        def __init__(self, model_name="efficientnet_b0", n_classes=len(CLASS_NAMES), pretrained=False, dropout=0.3):
            super().__init__()
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            feat = self.backbone.num_features
            self.head = torch.nn.Sequential(
                torch.nn.Linear(feat, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(512, n_classes)
            )
        def forward(self, x):
            x = self.backbone(x)
            x = self.head(x)
            return x

    pt_model = PTWrapper(model_name="efficientnet_b0", n_classes=len(CLASS_NAMES), pretrained=False, dropout=0.3)
    state = torch.load(str(PYTORCH_MODEL_PATH), map_location=DEVICE)
    # if saved with checkpoint dict:
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    pt_model.load_state_dict(state)
    pt_model.to(DEVICE)
    pt_model.eval()
    print("Loaded PyTorch model.")
except Exception as e:
    print("Warning: PyTorch model load failed:", e)
    pt_model = None

# 3) ONNX Runtime model
onnx_sess = None
try:
    onnx_sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    # if GPU available and onnxruntime-gpu installed:
    # onnx_sess = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CUDAExecutionProvider"])
    print("Loaded ONNX model.")
except Exception as e:
    print("Warning: ONNX model load failed:", e)
    onnx_sess = None

AVAILABLE_MODELS = {
    "tf": tf_model is not None,
    "pt": pt_model is not None,
    "onnx": onnx_sess is not None
}
print("Available models:", AVAILABLE_MODELS)

# ---------------------------
# Inference wrappers
# ---------------------------
def predict_tf(img_np: np.ndarray) -> np.ndarray:
    """Run TF model => returning softmax probabilities as 1D numpy array."""
    if tf_model is None:
        raise RuntimeError("TF model not loaded")
    x = prepare_for_tf(img_np)  # NHWC
    probs = tf_model(x, training=False).numpy()
    probs = softmax_numpy(probs[0])
    return probs

def predict_pt(img_np: np.ndarray) -> np.ndarray:
    if pt_model is None:
        raise RuntimeError("PyTorch model not loaded")
    x = prepare_for_torch(img_np)  # NCHW torch tensor
    with torch.no_grad():
        outputs = pt_model(x)
        probs = F.softmax(outputs, dim=1).detach().cpu().numpy()[0]
    return probs

def predict_onnx(img_np: np.ndarray) -> np.ndarray:
    if onnx_sess is None:
        raise RuntimeError("ONNX model not loaded")
    x = prepare_for_onnx(img_np)  # NCHW numpy
    # find input name
    input_name = onnx_sess.get_inputs()[0].name
    out_name = onnx_sess.get_outputs()[0].name
    raw = onnx_sess.run([out_name], {input_name: x})
    logits = raw[0][0]
    probs = softmax_numpy(logits)
    return probs

def softmax_numpy(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()

# ---------------------------
# Ensemble logic
# ---------------------------
def ensemble_predict(probs_list: List[np.ndarray], class_names: List[str]) -> Dict:
    """
    probs_list: list of 1D arrays from each model (length = n_classes)
    returns dict with per-model top predictions and ensemble summary.
    """
    # stack and check shapes
    probs_stack = np.vstack(probs_list)  # shape (n_models, n_classes)
    avg_probs = probs_stack.mean(axis=0)
    ensemble_class_idx = int(np.argmax(avg_probs))
    ensemble_class = class_names[ensemble_class_idx]
    ensemble_confidence = float(avg_probs[ensemble_class_idx])  # [0,1], average prob for predicted class

    # per-model top classes
    per_model_preds = []
    for probs in probs_list:
        idx = int(np.argmax(probs))
        per_model_preds.append({
            "class": class_names[idx],
            "class_idx": idx,
            "prob": float(probs[idx]),
            "all_probs": probs.tolist()
        })

    # agreement score: fraction of models that predicted the ensemble_class
    votes = [1 if p["class_idx"] == ensemble_class_idx else 0 for p in per_model_preds]
    agreement = float(sum(votes) / len(votes))

    # simple calibrated confidence combining agreement and avg_probs:
    # final_confidence = ensemble_confidence * (0.6 + 0.4 * agreement)
    final_confidence = float(ensemble_confidence * (0.6 + 0.4 * agreement))

    return {
        "ensemble_class": ensemble_class,
        "ensemble_class_idx": ensemble_class_idx,
        "ensemble_avg_probs": avg_probs.tolist(),
        "ensemble_confidence_raw": ensemble_confidence,
        "agreement": agreement,
        "final_confidence": final_confidence,
        "per_model": per_model_preds
    }

# ---------------------------
# Advice mapping - customize for 25 classes
# ---------------------------
def get_advice(condition_name):
    """Generate advice based on condition name with intelligent fallback"""
    
    # High urgency conditions (malignant/serious)
    high_urgency_keywords = [
        "malignant", "melanoma", "carcinoma", "cancer", "lesion"
    ]
    
    # Medium urgency (infectious, inflammatory)
    medium_urgency_keywords = [
        "infection", "bacterial", "fungal", "viral", "herpes", "hiv", "std",
        "cellulitis", "impetigo", "vasculitis", "lupus"
    ]
    
    # Low urgency (benign, cosmetic)
    low_urgency_keywords = [
        "benign", "healthy", "keratoses", "hair loss", "alopecia",
        "pigmentation", "vitiligo"
    ]
    
    condition_lower = condition_name.lower()
    
    # Check for high urgency
    if any(keyword in condition_lower for keyword in high_urgency_keywords):
        return {
            "short": f"Possible {condition_name}. Immediate dermatologist consultation recommended for evaluation and possible biopsy.",
            "urgency": "high",
            "recommendation": "Schedule urgent dermatology appointment"
        }
    
    # Check for medium urgency
    elif any(keyword in condition_lower for keyword in medium_urgency_keywords):
        return {
            "short": f"{condition_name} detected. Medical evaluation recommended for diagnosis and treatment.",
            "urgency": "medium",
            "recommendation": "Schedule dermatology appointment within 1-2 weeks"
        }
    
    # Check for healthy skin
    elif "healthy" in condition_lower:
        return {
            "short": "Skin appears healthy. Continue regular skin checks and sun protection.",
            "urgency": "low",
            "recommendation": "Maintain routine skin care and monitoring"
        }
    
    # Low urgency or general case
    else:
        return {
            "short": f"Possible {condition_name}. Consider dermatology consultation for proper diagnosis and management.",
            "urgency": "medium",
            "recommendation": "Schedule dermatology appointment if symptoms persist or worsen"
        }


# Specific advice for known conditions (optional override)
SPECIFIC_ADVICE = {
    "Healthy_Skin": {
        "short": "Skin appears healthy. Continue regular monitoring and sun protection.",
        "urgency": "low",
        "recommendation": "Maintain good skin care routine"
    },
    "Melanoma Skin Cancer Nevi and Moles": {
        "short": "URGENT: Possible melanoma or suspicious mole. Immediate dermatologist evaluation required.",
        "urgency": "high",
        "recommendation": "Schedule URGENT dermatology appointment for biopsy"
    },
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": {
        "short": "URGENT: Possible malignant lesion. Immediate dermatologist evaluation required.",
        "urgency": "high",
        "recommendation": "Schedule URGENT dermatology appointment"
    },
    "vitiligo": {
        "short": "Vitiligo detected - a skin condition causing loss of pigmentation. Consult dermatologist for treatment options.",
        "urgency": "low",
        "recommendation": "Schedule dermatology appointment for management options"
    }
}

# ---------------------------
# Routes
# ---------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "models": AVAILABLE_MODELS})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts: multipart form with 'file' field containing the image.
    Returns JSON with per-model predictions and ensemble.
    """
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400
    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": "invalid image", "detail": str(e)}), 400

    img_np = pil_to_numpy(image, size=IMG_SIZE)

    probs_list = []
    used_models = []
    errors = {}
    # Try TF
    try:
        if tf_model is not None:
            p = predict_tf(img_np)
            probs_list.append(p)
            used_models.append("tf")
    except Exception as e:
        errors["tf"] = str(e)
    # PyTorch
    try:
        if pt_model is not None:
            p = predict_pt(img_np)
            probs_list.append(p)
            used_models.append("pt")
    except Exception as e:
        errors["pt"] = str(e)
    # ONNX
    try:
        if onnx_sess is not None:
            p = predict_onnx(img_np)
            probs_list.append(p)
            used_models.append("onnx")
    except Exception as e:
        errors["onnx"] = str(e)

    if len(probs_list) == 0:
        return jsonify({"error": "no models available", "detail": errors}), 500

    # Build ensemble
    ensemble = ensemble_predict(probs_list, CLASS_NAMES)

    # Advice - use specific advice if available, otherwise generate intelligent advice
    label = ensemble["ensemble_class"]
    if label in SPECIFIC_ADVICE:
        advice = SPECIFIC_ADVICE[label]
    else:
        advice = get_advice(label)

    response = {
        "prediction": label,
        "prediction_idx": ensemble["ensemble_class_idx"],
        "final_confidence": ensemble["final_confidence"],
        "agreement": ensemble["agreement"],
        "ensemble_avg_probs": ensemble["ensemble_avg_probs"],
        "per_model_predictions": ensemble["per_model"],
        "advice": advice,
        "errors": errors
    }

    # Medical disclaimer (must be surfaced on client)
    response["disclaimer"] = ("This tool is for informational/demo purposes only. "
                              "Not a medical diagnosis. Seek a qualified clinician for medical advice.")

    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)


