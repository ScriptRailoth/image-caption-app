import streamlit as st
from PIL import Image
import torch
from collections import defaultdict
import spacy
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer

nlp = spacy.load("en_core_web_sm")

labels = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding"
]

label_synonyms = {
    "Enlarged Cardiomediastinum": ["cardiomediastinal enlargement", "enlargement of the cardiac silhouette", "shift of the mediastinal structures"],
    "Cardiomegaly": ["enlarged heart", "moderate cardiomegaly", "heart size is normal"],
    "Lung Opacity": ["lung densities", "pulmonary opacities"],
    "Lung Lesion": ["pulmonary lesion"],
    "Edema": ["fluid retention", "pulmonary vascular congestion", "vascular engorgement"],
    "Consolidation": ["pulmonary consolidation", "air-space opacity", "consolidation"],
    "Pneumonia": ["lung infection", "no evidence of pneumonia"],
    "Atelectasis": ["partial lung collapse", "atelectatic changes"],
    "Pneumothorax": ["collapsed lung", "no definite pneumothorax"],
    "Pleural Effusion": ["pleural fluid", "pleural thickening"],
    "Pleural Other": ["pleural abnormalities"],
    "Fracture": ["bone fractures"],
    "Support Devices": ["endotracheal tube", "central catheter"],
    "No Finding": ["lungs are clear", "normal", "unremarkable"]
}

def extract_clauses(sent):
    doc = nlp(sent)
    return [" ".join([t.text for t in token.subtree]) for token in doc if token.dep_ == "ROOT"]

def extract_observations(report):
    try:
        doc = nlp(report)
    except:
        return {label: "No relevant information" for label in labels}

    report_info = defaultdict(str)
    for sent in doc.sents:
        for label in labels:
            sent_lower = sent.text.lower()
            if label.lower() in sent_lower or any(syn in sent_lower for syn in label_synonyms.get(label, [])):
                clauses = extract_clauses(sent.text)
                report_info[label] += " ".join(clauses) + " " if clauses else sent.text + " "

    return {label: report_info[label].strip() or "No relevant information" for label in labels}

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("model/CXR_model")
    feature_extractor = ViTFeatureExtractor.from_pretrained("model/CXR_model")
    tokenizer = GPT2Tokenizer.from_pretrained("model/CXR_model")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.num_beams = 4
    model.eval()
    return model, feature_extractor, tokenizer

model, feature_extractor, tokenizer = load_model()

st.title("🩻 Chest X-ray Caption Generator with Report Structure")

uploaded_file = st.file_uploader("Upload a Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_length=50,
            num_beams=4,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=tokenizer.bos_token_id,
        )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    st.markdown("### 📝 Generated Caption:")
    st.success(caption)

    structured_report = extract_observations(caption)

    st.markdown("### 📋 Structured Observations:")
    for label in labels:
        st.write(f"- **{label}**: {structured_report[label]}")
