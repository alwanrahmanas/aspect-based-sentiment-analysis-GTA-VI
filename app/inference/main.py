from aspect_extraction.model import PredictAspect
from sentiment_classification.model import PredictSentiment
import pandas as pd
from time import time

# ===== Contoh pemakaian =====

if __name__ == "__main__":
    aspect_model = PredictAspect()
    sentiment_model = PredictSentiment()

    texts = [
    "I would watch the trailer for another 100 years. The vibes it shows reminds me about my childhood era.",
    "Such a nostalgic moment",
    ]

    time_start = time()

    # Step 1: Aspect Extraction
    df_aspects = aspect_model.predict(texts)
    print("Aspect Extraction Results:")
    print(df_aspects)

    # Step 2: Sentiment Classification
    df_sentiments = sentiment_model.predict(df_aspects)
    print("\nSentiment Classification Results:")
    print(df_sentiments)

    time_end = time()
    print(f"\nTotal processing time: {time_end - time_start:.2f} seconds")