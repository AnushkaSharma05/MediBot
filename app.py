# app.py
import os
# Workaround for macOS OpenMP runtime conflicts (common with PyTorch + Conda/Brew).
# Must be set before importing torch (directly or indirectly).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from unittest import result

from flask import Flask, request, jsonify, render_template
import sys
sys.path.append('.')

from src.inference import load_model_and_vocab, predict
import config

from flask import Flask, request, jsonify, render_template, session
import secrets


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# ── Load model ONCE at startup ─────────────────────────
print("[App] Loading model...")
model, vocab = load_model_and_vocab()
print("[App] ✅ Ready!")

def get_severity_level(severity_score):
    """
    Convert numeric severity to level + advice.
    Severity weights from dataset range roughly 1-7.
    """
    if severity_score <= 3:
        return {
            "level"  : "mild",
            "emoji"  : "🟢",
            "label"  : "Mild",
            "advice" : "You can monitor your symptoms at home. Stay hydrated and rest.",
            "urgent" : False
        }
    elif severity_score <= 5:
        return {
            "level"  : "moderate",
            "emoji"  : "🟡",
            "label"  : "Moderate",
            "advice" : "Consider consulting a doctor if symptoms persist or worsen.",
            "urgent" : False
        }
    else:
        return {
            "level"  : "severe",
            "emoji"  : "🔴",
            "label"  : "Severe",
            "advice" : "Please seek immediate medical attention!",
            "urgent" : True
        }


@app.route("/")
def index():
    """Serve the chat UI"""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    # ── Handle reset command ───────────────────────────
    if message.lower() in ["reset", "clear", "start over", "new"]:
        session.pop("symptoms", None)
        return jsonify({
            "type"   : "reset",
            "message": "✅ Conversation reset! Tell me your symptoms again."
        })

    # ── Get or init symptom memory ─────────────────────
    remembered_symptoms = session.get("symptoms", [])

    # ── Run inference on current message ──────────────
    result = predict(message, model, vocab)

    if result is None:
        return jsonify({
            "type"   : "error",
            "message": "Something went wrong. Please try again."
        })

    # ── Merge new symptoms with remembered ones ────────
    new_symptoms = result.get("symptoms_matched", [])
    all_symptoms = list(dict.fromkeys(
        remembered_symptoms + new_symptoms  # deduplicated
    ))

    if not all_symptoms:
        return jsonify({
            "type"   : "no_match",
            "message": "😕 I couldn't identify any known symptoms.\n\nTry: 'I have itching, skin rash and fever'"
        })

    # ── Re-run prediction with ALL accumulated symptoms
    import torch
    vector = vocab.symptoms_to_vector(all_symptoms)
    X      = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1)

    confidence, pred_idx = probs.max(dim=1)
    confidence = confidence.item()
    pred_idx   = pred_idx.item()
    disease    = vocab.idx2disease[pred_idx]

    top3_probs, top3_idx = probs[0].topk(3)
    top3 = [
        (vocab.idx2disease[i.item()], round(p.item(), 4))
        for p, i in zip(top3_probs, top3_idx)
    ]

    # ── Save accumulated symptoms to session ──────────
    session["symptoms"] = all_symptoms

    # ── Fetch description + precautions ───────────────
    desc_row = vocab.desc_df[
        vocab.desc_df["Disease"].str.strip() == disease
    ]
    description = (
        desc_row["Description"].values[0]
        if len(desc_row) > 0 else "No description available."
    )

    prec_row = vocab.prec_df[
        vocab.prec_df["Disease"].str.strip() == disease
    ]
    precautions = []
    if len(prec_row) > 0:
        for col in ["Precaution_1","Precaution_2",
                    "Precaution_3","Precaution_4"]:
            val = prec_row[col].values[0]
            if val and str(val) != "nan":
                precautions.append(str(val))

    # ── Severity ───────────────────────────────────────
    sev_col        = vocab.sev_df.columns[0]
    total_severity = 0
    for sym in all_symptoms:
        sev_row = vocab.sev_df[vocab.sev_df[sev_col] == sym]
        if len(sev_row) > 0:
            total_severity += int(sev_row["weight"].values[0])
    avg_severity  = round(total_severity / len(all_symptoms), 2)
    severity_info = get_severity_level(avg_severity)

    # ── Was anything new added this message? ──────────
    newly_added = [s for s in new_symptoms
                   if s not in remembered_symptoms]
    
    # ── Require at least 2 symptoms ───────────────────────
    if len(all_symptoms) < 2:
        return jsonify({
        "type"   : "low_confidence",
        "message": "I need at least 2 symptoms to make a reliable diagnosis.",
        "disease"       : disease,
        "confidence"    : confidence,
        "top3"          : top3,
        "all_symptoms"  : all_symptoms,
        "newly_added"   : newly_added
    })

    # ── Low confidence → ask for more ─────────────────
    if confidence < config.CONFIDENCE_THRESHOLD:
        return jsonify({
            "type"          : "low_confidence",
            "message"       : f"I'm {confidence:.0%} confident. Could you describe more symptoms?",
            "disease"       : disease,
            "confidence"    : confidence,
            "top3"          : top3,
            "all_symptoms"  : all_symptoms,
            "newly_added"   : newly_added
        })

    # ── High confidence ────────────────────────────────
    return jsonify({
        "type"            : "prediction",
        "disease"         : disease,
        "confidence"      : f"{confidence:.0%}",
        "description"     : description,
        "precautions"     : precautions,
        "severity"        : avg_severity,
        "severity_info"   : severity_info,
        "symptoms_matched": all_symptoms,       # ALL symptoms
        "newly_added"     : newly_added,        # just this turn
        "top3"            : top3
    })


@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    """Return all known symptoms — useful for autocomplete later"""
    return jsonify({
        "symptoms": sorted(vocab.all_symptoms)
    })


if __name__ == "__main__":
    app.run(debug=True)