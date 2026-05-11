# 🏥 Medical Diagnosis Chatbot

An end-to-end NLP-powered medical diagnosis chatbot built with 
PyTorch, Flask, and NLP techniques. Users describe symptoms in 
natural language and receive disease predictions with descriptions,
severity assessments, and precautions.

> ⚠️ **Disclaimer:** This project is for educational purposes only.
> Always consult a qualified medical professional for real diagnosis.

---

## � Live Demo

Try the chatbot live: **[https://mediscan-ayxg.onrender.com/](https://mediscan-ayxg.onrender.com/)**

---

## �🎯 Features

- **Natural Language Input** — type symptoms conversationally
- **Fuzzy Symptom Matching** — handles typos and alternate phrasings
- **Symptom Aliases** — maps "fever" → `high_fever`, "dizzy" → `dizziness`
- **Conversation Memory** — accumulates symptoms across messages for better accuracy
- **Confidence Thresholding** — asks for more symptoms when uncertain (<75%)
- **Severity Warning System** — color-coded urgency levels (Mild / Moderate / Severe)
- **Top 3 Predictions** — shows alternative diagnoses with probabilities
- **Disease Descriptions** — detailed explanation of predicted condition
- **Precautions** — actionable steps based on predicted disease

---

## 🏗️ Architecture
User Input (natural language)
↓
NLP Preprocessing
(tokenization, stop word removal, alias mapping)
↓
Fuzzy Symptom Extractor
(uni/bi/trigram matching via difflib)
↓
Bag-of-Symptoms Vector (131-dim binary)
↓
Feedforward Neural Network (PyTorch)
131 → 128 → 64 → 41
↓
Softmax → Disease Probabilities
↓
Confidence Check + Severity Assessment
↓
Flask API → Chat UI Response

---

## 📁 Project Structure
'''bash
medical_chatbot/
│
├── data/
│   ├── dataset.csv               # 4920 rows, 41 diseases, 131 symptoms
│   ├── symptom_Description.csv   # Disease descriptions
│   ├── symptom_severity.csv      # Symptom severity weights
│   └── symptom_precaution.csv    # Precautions per disease
│
├── src/
│   ├── preprocess.py             # Data loading and cleaning
│   ├── vocabulary.py             # Symptom/disease mappings + BoW
│   ├── dataset.py                # PyTorch Dataset class
│   ├── model.py                  # Neural network architecture
│   ├── train.py                  # Training pipeline
│   └── inference.py              # Prediction + NLP pipeline
│
├── model/
│   ├── model.pth                 # Trained model weights
│   └── model_data.pkl            # Saved vocabulary + mappings
│
├── templates/
│   └── index.html                # Chat UI
│
├── app.py                        # Flask backend
├── config.py                     # Hyperparameters and paths
├── requirements.txt
└── README.md
'''
---

## 🧠 Model Architecture
Input Layer    : 131 neurons  (one per unique symptom)
Hidden Layer 1 : 128 neurons  + BatchNorm + ReLU + Dropout(0.3)
Hidden Layer 2 : 64  neurons  + BatchNorm + ReLU + Dropout(0.3)
Output Layer   : 41  neurons  (one per disease)

| Property | Value |
|---|---|
| Total Parameters | 28,201 |
| Training Samples | 3,936 (80%) |
| Validation Samples | 984 (20%) |
| Final Val Accuracy | 100% |
| Optimizer | Adam (lr=0.001) |
| Loss Function | CrossEntropyLoss |
| Epochs | 300 |

---

## 📊 Dataset

**Source:** [Kaggle]

| File | Records | Description |
|---|---|---|
| `dataset.csv` | 4,920 rows | Disease-symptom mappings |
| `symptom_Description.csv` | 41 diseases | Disease descriptions |
| `symptom_severity.csv` | 133 symptoms | Severity weights (1-7) |
| `symptom_precaution.csv` | 41 diseases | 4 precautions each |

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### 2. Create virtual environment
```bash
python -m venv medbot_env
source medbot_env/bin/activate      # Mac/Linux
medbot_env\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add dataset files
Download from Kaggle and place all 4 CSV files in `data/`

### 5. Train the model
```bash
python src/train.py
```

### 6. Run the chatbot
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 💬 Example Conversations

**High confidence:**
User : I have itching, skin rash and nodal eruptions
Bot  : 🔍 Fungal Infection — 100% confidence
🟢 Mild Severity
✅ Precautions: bath twice, use neem water...

**Low confidence → memory accumulation:**
User : I have stomach pain and vomiting
Bot  : 🤔 67% confident — need more symptoms
User : also acidity and cough
Bot  : 🔍 GERD — 100% confidence
🟡 Moderate Severity

**Severe case:**
User : I have chest pain, breathlessness and sweating
Bot  : 🔍 Heart Attack — 100% confidence
🚨 Please seek immediate medical attention!
🔴 Severe Severity — 5.5/7

---

## 🔧 NLP Techniques Used

- **Tokenization** — split user input into tokens
- **Stop Word Removal** — filter filler words
- **N-gram Matching** — uni/bi/trigram symptom extraction
- **Fuzzy Matching** — `difflib.get_close_matches` for typo handling
- **Symptom Aliasing** — maps colloquial terms to medical vocabulary
- **Bag of Symptoms** — 131-dimensional binary feature vector
- **Confidence Calibration** — threshold-based response routing
- **Session Memory** — Flask sessions for multi-turn accumulation

---

## 🔮 Possible Upgrades

- [ ] Word2Vec / FastText symptom embeddings
- [ ] Symptom autocomplete dropdown
- [ ] Multi-language support
- [ ] User feedback collection + model retraining
- [ ] REST API with authentication
- [ ] Docker containerization

---

## 👩‍💻 Author

Anushka Sharma
Built as an end-to-end NLP + Deep Learning portfolio project.
