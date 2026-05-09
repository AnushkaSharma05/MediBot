# src/model.py
import os
# Workaround for macOS OpenMP runtime conflicts (PyTorch aborts with exit code 134).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class MedicalDiagnosisNet(nn.Module):
    """
    Feedforward Neural Network for disease classification.

    Architecture:
        Input(131) → FC(128) → BN → ReLU → Dropout
                   → FC(64)  → BN → ReLU → Dropout
                   → FC(41)  → Output logits

    Input  : float tensor of shape (batch_size, 131)
    Output : float tensor of shape (batch_size, 41)  ← raw logits
    """

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MedicalDiagnosisNet, self).__init__()

        # ── Layer 1: 131 → 128 ────────────────────────
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 131 → 128
            nn.BatchNorm1d(hidden_size),          # normalize activations
            nn.ReLU(),                            # non-linearity
            nn.Dropout(dropout)                   # regularization
        )

        # ── Layer 2: 128 → 64 ─────────────────────────
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 128 → 64
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Output Layer: 64 → 41 ─────────────────────
        # No activation here — CrossEntropyLoss expects raw logits
        self.output = nn.Linear(hidden_size // 2, output_size)  # 64 → 41

    def forward(self, x):
        """
        Forward pass through the network.

        x shape : (batch_size, 131)
        returns : (batch_size, 41) raw logits
        """
        x = self.layer1(x)   # (B, 131) → (B, 128)
        x = self.layer2(x)   # (B, 128) → (B, 64)
        x = self.output(x)   # (B, 64)  → (B, 41)
        return x             # raw logits — no softmax here


def build_model(vocab):
    """
    Factory function — builds model with correct sizes from vocab.
    Always use this instead of constructing manually.
    """
    model = MedicalDiagnosisNet(
        input_size  = vocab.n_symptoms,   # 131
        hidden_size = config.HIDDEN_SIZE, # 128
        output_size = vocab.n_diseases,   # 41
        dropout     = config.DROPOUT      # 0.3
    )
    return model


# ── Quick test ──────────────────────────────────────────
if __name__ == "__main__":
    from src.vocabulary import MedicalVocabulary

    vocab = MedicalVocabulary()
    model = build_model(vocab)

    # Print full architecture
    print("\n── Model Architecture ──────────────────────")
    print(model)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print(f"\n── Parameter Count ─────────────────────────")
    print(f"Total trainable params : {total_params:,}")

    # ── Dry run with fake batch ──────────────────────
    print(f"\n── Dry Run (fake batch) ────────────────────")
    fake_batch = torch.zeros(32, vocab.n_symptoms)  # (32, 131)
    fake_batch[0][5]  = 1.0  # activate symptom 5
    fake_batch[0][10] = 1.0  # activate symptom 10

    model.eval()  # turn off dropout for testing
    with torch.no_grad():
        output = model(fake_batch)  # (32, 41)

    print(f"Input  shape : {fake_batch.shape}")  # [32, 131]
    print(f"Output shape : {output.shape}")       # [32, 41]

    # Convert one output to probabilities
    probs = torch.softmax(output[0], dim=0)
    print(f"\nFirst sample probabilities:")
    print(f"  Shape : {probs.shape}")             # [41]
    print(f"  Sum   : {probs.sum():.4f}")         # should be 1.0000
    print(f"  Max   : {probs.max():.4f}")
    print(f"  Predicted disease index: {probs.argmax().item()}")
    print(f"\n✅ Model working correctly!")