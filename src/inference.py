
# src/inference.py
import torch
import pickle
import difflib
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.model import build_model

FUZZY_THRESHOLD = 0.75

# Common user words → exact dataset symptom names
SYMPTOM_ALIASES = {
    "fever"        : "high_fever",
    "temperature"  : "high_fever",
    "cold"         : "shivering",
    "tired"        : "fatigue",
    "tiredness"    : "fatigue",
    "exhausted"    : "fatigue",
    "throwup"      : "vomiting",
    "throw up"     : "vomiting",
    "puke"         : "vomiting",
    "poo"          : "diarrhoea",
    "diarrhea"     : "diarrhoea",
    "tummy"        : "stomach_pain",
    "belly"        : "stomach_pain",
    "sore_throat"  : "throat_irritation",
    "runny_nose"   : "runny_nose",
    "eye_watering" : "watering_from_eyes",
    "spots"        : "skin_rash",
    "redness"      : "skin_rash",
    "breathless"   : "breathlessness",
    "short_breath" : "breathlessness",
    "chest_tight"  : "chest_pain",
    "dizzy"        : "dizziness",
    "dizziness"    : "dizziness",
    "fit"          : "seizures",
    "fits"         : "seizures",
    "swollen"      : "swollen_legs",
    "pee"          : "burning_micturition",
    "urine_burn"   : "burning_micturition",
    "yellow_skin"  : "yellowing_of_eyes",
    "yellow_eyes"  : "yellowing_of_eyes",
}


def load_model_and_vocab():
    with open(config.DATA_SAVE_PATH, "rb") as f:
        vocab = pickle.load(f)

    model = build_model(vocab)
    model.load_state_dict(
        torch.load(config.MODEL_SAVE_PATH,
                   map_location=torch.device("cpu"))
    )
    model.eval()
    print("[Inference] ✅ Model and vocab loaded successfully!")
    return model, vocab


def fuzzy_match_symptom(user_symptom, known_symptoms):
    user_symptom = user_symptom.strip().lower().replace(" ", "_")

    # ── Check aliases first ────────────────────────────
    if user_symptom in SYMPTOM_ALIASES:
        return SYMPTOM_ALIASES[user_symptom]

    # ── Exact match ────────────────────────────────────
    if user_symptom in known_symptoms:
        return user_symptom

    # ── Fuzzy match ────────────────────────────────────
    matches = difflib.get_close_matches(
        user_symptom,
        known_symptoms,
        n=1,
        cutoff=FUZZY_THRESHOLD
    )
    return matches[0] if matches else None


def extract_symptoms(user_text, known_symptoms):
    text = user_text.lower().strip()

    stop_words = {
        "i", "have", "had", "been", "am", "is", "are",
        "the", "a", "an", "and", "or", "with", "also",
        "my", "me", "feeling", "feel", "experiencing",
        "please", "help", "suffering", "since", "some"
    }

    tokens = [t.strip(".,!?") for t in text.split()]
    tokens = [t for t in tokens if t and t not in stop_words]

    matched      = []
    unmatched    = []
    used_indices = set()

    # ── Trigrams ───────────────────────────────────────
    for i in range(len(tokens) - 2):
        if i in used_indices:
            continue
        candidate = "_".join(tokens[i:i+3])
        match = fuzzy_match_symptom(candidate, known_symptoms)
        if match and match not in matched:
            matched.append(match)
            used_indices.update([i, i+1, i+2])

    # ── Bigrams ────────────────────────────────────────
    for i in range(len(tokens) - 1):
        if i in used_indices or i+1 in used_indices:
            continue                        # ← KEY FIX
        candidate = "_".join(tokens[i:i+2])
        match = fuzzy_match_symptom(candidate, known_symptoms)
        if match and match not in matched:
            matched.append(match)
            used_indices.update([i, i+1])

    # ── Unigrams ───────────────────────────────────────
    for i, token in enumerate(tokens):
        if i in used_indices:
            continue
        match = fuzzy_match_symptom(token, known_symptoms)
        if match and match not in matched:
            matched.append(match)
            used_indices.add(i)
        else:
            if token not in matched:
                unmatched.append(token)

    return matched, unmatched


def predict(user_text, model, vocab):
    try:
        # ── Step 1: Extract symptoms ───────────────────
        matched, unmatched = extract_symptoms(
            user_text, vocab.all_symptoms
        )

        # ── Step 2: No symptoms found ──────────────────
        if not matched:
            return {
                "disease"         : None,
                "confidence"      : 0.0,
                "symptoms_matched": [],
                "symptoms_unknown": unmatched,
                "low_confidence"  : True,
                "message"         : "No recognizable symptoms found."
            }

        # ── Step 3: Build input vector ─────────────────
        vector = vocab.symptoms_to_vector(matched)
        X = torch.tensor(
            vector, dtype=torch.float32
        ).unsqueeze(0)              # shape: (1, 131)

        # ── Step 4: Forward pass ───────────────────────
        with torch.no_grad():
            logits = model(X)                      # (1, 41)
            probs  = torch.softmax(logits, dim=1)  # (1, 41)

        confidence, pred_idx = probs.max(dim=1)
        confidence = confidence.item()
        pred_idx   = pred_idx.item()
        disease    = vocab.idx2disease[pred_idx]

        # ── Step 5: Top 3 predictions ──────────────────
        top3_probs, top3_idx = probs[0].topk(3)
        top3 = [
            (vocab.idx2disease[i.item()], round(p.item(), 4))
            for p, i in zip(top3_probs, top3_idx)
        ]

        # ── Step 6: Description ────────────────────────
        desc_row = vocab.desc_df[
            vocab.desc_df["Disease"].str.strip() == disease
        ]
        description = (
            desc_row["Description"].values[0]
            if len(desc_row) > 0
            else "No description available."
        )

        # ── Step 7: Precautions ────────────────────────
        prec_row = vocab.prec_df[
            vocab.prec_df["Disease"].str.strip() == disease
        ]
        precautions = []
        if len(prec_row) > 0:
            for col in ["Precaution_1", "Precaution_2",
                        "Precaution_3", "Precaution_4"]:
                val = prec_row[col].values[0]
                if val and str(val) != "nan":
                    precautions.append(str(val))

        # ── Step 8: Severity score ─────────────────────
        sev_col        = vocab.sev_df.columns[0]
        total_severity = 0
        for sym in matched:
            sev_row = vocab.sev_df[
                vocab.sev_df[sev_col] == sym
            ]
            if len(sev_row) > 0:
                total_severity += int(sev_row["weight"].values[0])

        avg_severity = round(total_severity / len(matched), 2)

        return {
            "disease"         : disease,
            "confidence"      : round(confidence, 4),
            "symptoms_matched": matched,
            "symptoms_unknown": unmatched,
            "description"     : description,
            "precautions"     : precautions,
            "severity"        : avg_severity,
            "top3"            : top3,
            "low_confidence"  : confidence < config.CONFIDENCE_THRESHOLD
        }

    except Exception as e:
        # ── Now we'll actually SEE errors ─────────────
        print(f"[Inference] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    model, vocab = load_model_and_vocab()

    tests = [
        "I have itching and skin rash with nodal skin eruptions",
        "continuous sneezing shivering and chills",
        "stomach pain acidity vomiting and cough",
        "random words that mean nothing"
    ]

    for text in tests:
        print(f"\n{'='*55}")
        print(f"Input : {text}")
        result = predict(text, model, vocab)
        if result and result["disease"]:
            print(f"Disease    : {result['disease']}")
            print(f"Confidence : {result['confidence']:.2%}")
            print(f"Severity   : {result['severity']}")
            print(f"Top 3      : {result['top3']}")
            print(f"Precautions: {result['precautions'][:2]}")
        elif result:
            print(f"Result     : {result.get('message')}")