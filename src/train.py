# src/train.py
import os
# Workaround for macOS OpenMP runtime conflicts (PyTorch aborts with exit code 134).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.vocabulary import MedicalVocabulary
from src.dataset    import SymptomDataset
from src.model      import build_model


def train():
    print("\n" + "="*55)
    print("       MEDICAL CHATBOT — TRAINING PIPELINE")
    print("="*55)

    # ── 1. Load vocabulary + dataset ──────────────────
    vocab   = MedicalVocabulary()
    dataset = SymptomDataset(vocab)

    # ── 2. Train / Validation split (80/20) ───────────
    total      = len(dataset)
    val_size   = int(0.2 * total)
    train_size = total - val_size

    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible
    )

    train_loader = DataLoader(train_set,
                              batch_size=config.BATCH_SIZE,
                              shuffle=True)
    val_loader   = DataLoader(val_set,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False)

    print(f"\n[Data Split]")
    print(f"  Train samples : {train_size}")
    print(f"  Val   samples : {val_size}")

    # ── 3. Build model ─────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"\n[Device] Using : {device}")

    model     = build_model(vocab).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.LEARNING_RATE)

    # ── 4. Learning rate scheduler ────────────────────
    # Reduces LR by 0.5 if val_loss doesn't improve for 20 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=20
    )

    # ── 5. Training loop ───────────────────────────────
    print(f"\n[Training] Starting {config.EPOCHS} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | "
          f"{'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    best_val_loss = float('inf')
    best_epoch    = 0

    for epoch in range(1, config.EPOCHS + 1):

        # ── Train phase ────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()           # clear old gradients
            outputs = model(X_batch)        # forward pass → (B, 41)
            loss    = criterion(outputs, y_batch)  # compute loss
            loss.backward()                 # backward pass
            optimizer.step()               # update weights

            train_loss    += loss.item()
            preds          = outputs.argmax(dim=1)  # predicted class
            train_correct += (preds == y_batch).sum().item()
            train_total   += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = train_correct / train_total

        # ── Validation phase ───────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs    = model(X_batch)
                loss       = criterion(outputs, y_batch)
                val_loss  += loss.item()
                preds      = outputs.argmax(dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total   += y_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = val_correct / val_total

        # ── LR Scheduler step ─────────────────────────
        scheduler.step(avg_val_loss)

        # ── Save best model ───────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch    = epoch
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

        # ── Print every 10 epochs ─────────────────────
        if epoch % 10 == 0 or epoch == 1:
            print(f"{epoch:>6} | {avg_train_loss:>10.4f} | "
                  f"{train_acc:>8.2%} | {avg_val_loss:>8.4f} | "
                  f"{val_acc:>6.2%}")

    # ── 6. Save vocabulary ────────────────────────────
    vocab.save()

    print("-" * 55)
    print(f"\n✅ Training complete!")
    print(f"   Best epoch    : {best_epoch}")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Model saved   : {config.MODEL_SAVE_PATH}")
    print(f"   Vocab saved   : {config.DATA_SAVE_PATH}")


if __name__ == "__main__":
    start = time.time()
    train()
    elapsed = time.time() - start
    print(f"\n⏱  Total training time: {elapsed:.1f}s")