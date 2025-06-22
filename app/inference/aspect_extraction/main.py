from model import PredictAspect
import pandas as pd
# ===== Contoh pemakaian =====
if __name__ == "__main__":
    aspect_model = PredictAspect()
    texts = [
    "Bad trailer",
    "I would watch the trailer for another 100 years. The vibes it shows reminds me about my childhood era.",
    "Such a nostalgic moment"
    ]

    df_predictions = aspect_model.predict(texts)
    print(df_predictions)