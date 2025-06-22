import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

class PredictSentiment:
    def __init__(self,
                 model_name='alwanrahmana/sentiment-absa-model-cased',
                 label_map=None,
                 device=None):
        # Load tokenizer & model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
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
        token_type_ids = encoding['token_type_ids'].to(self.DEVICE)

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

