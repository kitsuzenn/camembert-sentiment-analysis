import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    labels = {0: "Négatif", 1: "Neutre", 2: "Positif"}
    return {labels[i]: float(probs[i]) for i in labels}

interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Entrez un texte en français"),
    outputs=gr.Label(label="Sentiment"),
    title="Analyse de Sentiment FR — CamemBERT 3 classes",
    description="Modèle CamemBERT fine-tuné sur un dataset custom 3 classes (Positif / Négatif / Neutre). Projet réalisé par Dahren.",
    examples=[
        ["Ce film est absolument génial, un chef-d'œuvre !"],
        ["Film nul, une perte de temps totale."],
        ["Le film dure 1h47 et sort en salles le 15 mars."],
    ]
)

interface.launch()
