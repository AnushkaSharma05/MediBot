# src/dataset.py
import os
# Workaround for macOS OpenMP runtime conflicts (PyTorch aborts with exit code 134).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import Dataset
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class SymptomDataset(Dataset):
    """
    PyTorch Dataset for disease-symptom data.

    Each sample:
        X → float tensor of shape (131,)  binary BoW vector
        y → long tensor scalar            disease index
    """

    def __init__(self, vocab):
        self.vocab   = vocab
        self.samples = []   # list of (vector, label_idx)
        self._build()

    def _build(self):
        """
        Convert every row in dataset.csv into (X, y) pair.
        Handles sparse symptom columns (NaN = no symptom).
        """
        df           = self.vocab.df
        symptom_cols = [c for c in df.columns if "Symptom" in c]
        skipped      = 0

        for _, row in df.iterrows():
            # ── Collect non-null symptoms for this row ──
            symptoms = []
            for col in symptom_cols:
                val = row[col]
                if val and str(val) != "nan":
                    symptoms.append(str(val).strip())

            # ── Skip rows with no valid symptoms ────────
            if not symptoms:
                skipped += 1
                continue

            # ── Build input vector ───────────────────────
            # shape: (131,) — binary float tensor
            X = torch.tensor(
                self.vocab.symptoms_to_vector(symptoms),
                dtype=torch.float32
            )

            # ── Build label ──────────────────────────────
            # single integer → disease index
            disease = row["Disease"].strip()
            y = torch.tensor(
                self.vocab.disease2idx[disease],
                dtype=torch.long
            )

            self.samples.append((X, y))

        print(f"[Dataset] ✅ Built {len(self.samples)} samples "
              f"(skipped {skipped} empty rows)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]  # returns (X, y) tuple


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    from src.vocabulary import MedicalVocabulary
    from torch.utils.data import DataLoader

    vocab   = MedicalVocabulary()
    dataset = SymptomDataset(vocab)

    print(f"\nDataset size     : {len(dataset)}")

    # Inspect first sample
    X, y = dataset[0]
    print(f"X shape          : {X.shape}")   # expect torch.Size([131])
    print(f"X dtype          : {X.dtype}")   # expect torch.float32
    print(f"y value          : {y}")          # expect tensor(int)
    print(f"y dtype          : {y.dtype}")   # expect torch.int64
    print(f"Active symptoms  : {int(X.sum())} symptoms in this row")
    print(f"Disease label    : {vocab.idx2disease[y.item()]}")

    # Test DataLoader with batch
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    Xb, yb = next(iter(loader))
    print(f"\nBatch X shape    : {Xb.shape}")  # expect [32, 131]
    print(f"Batch y shape    : {yb.shape}")   # expect [32]
    print(f"\n✅ DataLoader working correctly!")