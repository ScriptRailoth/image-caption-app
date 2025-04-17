
# 🩻 Chest X-ray Report Generator

This project builds an image captioning system for chest X-rays using a Vision Transformer encoder and GPT-2 decoder. It includes a web UI for uploading X-ray images and viewing structured diagnostic output.

---

## 📁 Project Structure

```
.
├── clean_radiology_reports.py       # Extracts CSVs from raw XML reports
├── CXR_model_training_final.ipynb   # Jupyter notebook to train captioning model
├── app.py                           # Streamlit app for image upload & prediction
├── model/                           # Folder for saved model + tokenizer + extractor
├── archive/                         # Folder containing images and raw reports
└── README.md
```

---

## ⚙️ Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, manually install:
```bash
pip install torch torchvision transformers pandas pillow scikit-learn tqdm spacy evaluate streamlit
python -m spacy download en_core_web_sm
```

2. **Prepare the dataset**

```bash
python clean_radiology_reports.py
```

This will generate:
- `indiana_reports.csv`
- `indiana_projections.csv`

---

## 🧠 Train the Model

1. Open `CXR_model_training_final.ipynb` in Jupyter.
2. Run all cells to:
   - Merge and preprocess data
   - Load pretrained ViT and GPT-2
   - Train the image captioning model
   - Save to `model/CXR_model/`

> You can also use the pretrained model directly by placing it in `model/CXR_model`.

---

## 🚀 Run the Streamlit App

```bash
streamlit run app.py
```

- Upload a chest X-ray image (`.jpg`, `.png`, etc.)
- The app will:
  - Generate a diagnostic-style caption
  - Break it into structured observations (e.g., “Cardiomegaly”, “Pneumothorax”)

---

## 📦 Notes

- Make sure the model folder exists at: `model/CXR_model/`
- Tested with Python 3.9+
