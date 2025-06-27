import os
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd


class PredictAspect(nn.Module):
    def __init__(self, 
                 model_name='alwanrahmana/aspect-detection-modernbert-base-logweight-FocalLoss',
                 num_labels=6,
                 threshold=0.30,
                 aspect_labels=None,
                 device=None):
        super(PredictAspect, self).__init__()

        # Path local simpanan model
        local_dir = f"./app/inference/pretrained_models/{model_name.replace('/', '_')}"
        
        # Check apakah model sudah ada di local
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"✅ Local model found at {local_dir}. Loading from local...")
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(local_dir, num_labels=num_labels)
        else:
            print(f"⬇️ Downloading model from Hugging Face Hub: {model_name} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

            os.makedirs(local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)
            print(f"✅ Model saved to {local_dir}")

        # Hyperparams & device config
        self.GLOBAL_THRESHOLD = threshold
        self.ASPECT_LABELS = aspect_labels if aspect_labels else [
            'Visual Graphic', 'Gameplay', 'Release Game', 'Plot and Character', 'Nostalgia', 'Trailer & Hype'
        ]
        self.DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.DEVICE)
        self.model.eval()

    def _predict_probs(self, texts, max_len=512):
        encoded_batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
            return_token_type_ids=False
        )

        input_ids = encoded_batch['input_ids'].to(self.DEVICE)
        attention_mask = encoded_batch['attention_mask'].to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

        return probs

    def predict(self, texts, max_len=512):
        """
        Full end-to-end pipeline:
        input: list of texts
        output: dataframe with detected aspects and probabilities
        """
        probs = self._predict_probs(texts, max_len=max_len)

        results = []
        for text, prob in zip(texts, probs):
            positive_aspects = []
            proba_values = []

            for aspect, p in zip(self.ASPECT_LABELS, prob):
                if p > self.GLOBAL_THRESHOLD:
                    positive_aspects.append(aspect)
                    proba_values.append(round(float(p), 4))

            results.append({
                'text': text,
                'aspects': positive_aspects,
                'proba_values': proba_values
            })

        df = pd.DataFrame(results)
        return df
