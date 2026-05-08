# config.py
import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODEL_DIR   = os.path.join(BASE_DIR, "model")

DATASET_PATH     = os.path.join(DATA_DIR, "dataset.csv")
DESCRIPTION_PATH = os.path.join(DATA_DIR, "symptom_Description.csv")
SEVERITY_PATH    = os.path.join(DATA_DIR, "symptom_severity.csv")
PRECAUTION_PATH  = os.path.join(DATA_DIR, "symptom_precaution.csv")

MODEL_SAVE_PATH  = os.path.join(MODEL_DIR, "model.pth")
DATA_SAVE_PATH   = os.path.join(MODEL_DIR, "model_data.pkl")

# ── Training Hyperparameters ───────────────────────────
HIDDEN_SIZE    = 128
LEARNING_RATE  = 0.001
BATCH_SIZE     = 32
EPOCHS         = 300
DROPOUT        = 0.3

# ── Inference ──────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75  