import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig


class PredictSentiment:
    def __init__(self,
                 model_name='alwanrahmana/sentiment-classification-deberta-base',
                 label_map=None,
                 device=None):
        # Local path buat nyimpan model
        local_dir = f"./app/inference/pretrained_models/{model_name.replace('/', '_')}"

        # Check apakah model sudah ada di local
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"✅ Local model found at {local_dir}. Loading from local...")
            config = AutoConfig.from_pretrained(local_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(local_dir, config=config)
        else:
            print(f"⬇️ Downloading model from Hugging Face Hub: {model_name} ...")
            config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

            os.makedirs(local_dir, exist_ok=True)
            self.tokenizer.save_pretrained(local_dir)
            self.model.save_pretrained(local_dir)
            print(f"✅ Model saved to {local_dir}")

        self.model.eval()

        # Device config
        self.DEVICE = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.DEVICE)

        # Label mapping
        self.label_map = label_map if label_map else {0: "Negative", 1: "Neutral", 2: "Positive"}

    def _predict_sentiment(self, aspect, text, max_len=256):
        encoding = self.tokenizer(
            aspect,
            text,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.DEVICE)
        attention_mask = encoding['attention_mask'].to(self.DEVICE)
        token_type_ids = encoding.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).item()

        return self.label_map.get(pred, "Unknown")

    def predict(self, df):
        """
        Input: DataFrame hasil PredictAspect
        Output: DataFrame dengan kolom text, aspect, predicted_sentiment
        """
        results = []

        for idx, row in df.iterrows():
            text = row['text']
            aspects = row['aspects']

            for aspect in aspects:
                sentiment = self._predict_sentiment(aspect, text)
                results.append({
                    'text': text,
                    'aspect': aspect,
                    'predicted_sentiment': sentiment
                })

        result_df = pd.DataFrame(results)
        return result_df
