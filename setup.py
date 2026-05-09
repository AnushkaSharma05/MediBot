# setup.py
import os, sys, requests
sys.path.append('.')

DATA_URLS = {
    "dataset.csv"             : "https://github.com/user-attachments/files/27555544/dataset.csv",
    "symptom_Description.csv" : "https://github.com/user-attachments/files/27555546/symptom_Description.csv",
    "symptom_precaution.csv"  : "https://github.com/user-attachments/files/27555548/symptom_precaution.csv",
    "symptom_severity.csv"    : "https://github.com/user-attachments/files/27555549/symptom_severity.csv",
}
def download_data():
    os.makedirs('data', exist_ok=True)
    for filename, url in DATA_URLS.items():
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            print(f"[Setup] ✅ {filename} already exists")
            continue
        print(f"[Setup] Downloading {filename}...")
        r = requests.get(url, allow_redirects=True)
        with open(filepath, 'wb') as f:
            f.write(r.content)
        print(f"[Setup] ✅ {filename} downloaded")

def train_if_needed():
    model_path = os.path.join('model', 'model.pth')
    vocab_path = os.path.join('model', 'model_data.pkl')
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print("[Setup] ✅ Model exists — skipping training")
        return
    print("[Setup] 🏋️  Training model...")
    os.makedirs('model', exist_ok=True)
    from src.train import train
    train()
    print("[Setup] ✅ Training complete!")

def setup():
    print("\n" + "="*50)
    print("     MEDISCAN AI — STARTUP SETUP")
    print("="*50)
    download_data()
    train_if_needed()
    print("="*50 + "\n")

if __name__ == "__main__":
    setup()