# src/vocabulary.py
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import load_and_analyze

class MedicalVocabulary:
    """
    Manages:
    - Master symptom list → index mapping
    - Disease → index mapping  
    - Converts symptom lists to BoW vectors
    """

    def __init__(self):
        # Load and analyze data
        (self.df, self.desc_df, 
         self.sev_df, self.prec_df,
         self.all_symptoms, 
         self.all_diseases) = load_and_analyze()

        # ── Core mappings ──────────────────────────────
        # symptom → index  (for building vectors)
        self.symptom2idx = {s: i for i, s 
                           in enumerate(self.all_symptoms)}
        
        # index → symptom  (for interpreting vectors)
        self.idx2symptom = {i: s for s, i 
                           in self.symptom2idx.items()}

        # disease → index  (for training labels)
        self.disease2idx = {d: i for i, d 
                           in enumerate(self.all_diseases)}
        
        # index → disease  (for interpreting predictions)
        self.idx2disease = {i: d for d, i 
                           in self.disease2idx.items()}

        # ── Sizes (critical for neural network) ────────
        self.n_symptoms = len(self.all_symptoms)   # input size  = 131
        self.n_diseases = len(self.all_diseases)   # output size = 41

        print(f"\n[Vocabulary] ✅ Built successfully!")
        print(f"  Input  size (symptoms) : {self.n_symptoms}")
        print(f"  Output size (diseases) : {self.n_diseases}")

    def symptoms_to_vector(self, symptom_list):
        """
        Convert a list of symptom strings → binary BoW vector

        Input : ["itching", "skin_rash", "fever"]
        Output: [0, 0, 1, 0, 1, ...] → length 131

        Unknown symptoms are silently ignored
        """
        vector = [0] * self.n_symptoms  # start with all zeros

        for symptom in symptom_list:
            symptom = symptom.strip().lower().replace(" ", "_")
            if symptom in self.symptom2idx:
                idx = self.symptom2idx[symptom]
                vector[idx] = 1             # mark as present
            else:
                print(f"  [Vocab] ⚠️  Unknown symptom: '{symptom}'")

        return vector  # length = 131

    def vector_to_symptoms(self, vector):
        """
        Convert binary vector back → symptom list
        Useful for debugging

        Input : [0, 0, 1, 0, 1, ...]
        Output: ["acidity", "back_pain", ...]
        """
        return [self.idx2symptom[i] 
                for i, val in enumerate(vector) if val == 1]

    def save(self):
        """Save vocabulary to disk for inference later"""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        with open(config.DATA_SAVE_PATH, "wb") as f:
            pickle.dump(self, f)
        print(f"[Vocabulary] 💾 Saved to {config.DATA_SAVE_PATH}")

    @staticmethod
    def load():
        """Load saved vocabulary from disk"""
        with open(config.DATA_SAVE_PATH, "rb") as f:
            return pickle.load(f)


# ── Quick test ─────────────────────────────────────────
if __name__ == "__main__":
    vocab = MedicalVocabulary()

    # Test symptom → vector → back to symptoms
    test_symptoms = ["itching", "skin_rash", "fever", "unknown_symptom"]
    print(f"\nTest input   : {test_symptoms}")
    
    vec = vocab.symptoms_to_vector(test_symptoms)
    print(f"Vector length: {len(vec)}")
    print(f"Active (1s)  : {sum(vec)}")  # should be 3 (unknown ignored)
    
    recovered = vocab.vector_to_symptoms(vec)
    print(f"Recovered    : {recovered}")

    # Show first 5 disease mappings
    print(f"\nDisease mappings (first 5):")
    for i in range(5):
        print(f"  {i} → {vocab.idx2disease[i]}")