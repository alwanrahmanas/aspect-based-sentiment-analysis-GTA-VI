import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

class PredictAspect(nn.Module):
    def __init__(self, 
                 model_name='alwanrahmana/aspect-detection-bert-large',
                 num_labels=6,
                 threshold=0.25,
                 aspect_labels=None,
                 device=None):
        super(PredictAspect, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Hyperparams & device config
        self.GLOBAL_THRESHOLD = threshold
        self.ASPECT_LABELS = aspect_labels if aspect_labels else ['Trailer & Hype',
                                                                  'Visual Graphic',
                                                                  'Plot and Character','Gameplay','Nostalgia','Rilis Game']
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


