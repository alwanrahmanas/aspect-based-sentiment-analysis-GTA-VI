from app.inference.aspect_extraction.model import PredictAspect
from app.inference.sentiment_classification.model import PredictSentiment
import pandas as pd
from time import time

if __name__ == "__main__":
    aspect_model = PredictAspect()
    sentiment_model = PredictSentiment()

    texts = [
        "I would watch the trailer for another 100 years. The vibes it shows reminds me about my childhood era.",
        "Such a nostalgic moment",
    ]

    time_start = time()
    df_aspects = aspect_model.predict(texts)
    time_after_aspect = time()

    df_sentiments = sentiment_model.predict(df_aspects)
    time_end = time()

    print("Aspect Extraction Results:")
    print(df_aspects)

    print("\nSentiment Classification Results:")
    print(df_sentiments)

    print(f"\nAspect Extraction time: {time_after_aspect - time_start:.2f} seconds")
    print(f"Sentiment Classification time: {time_end - time_after_aspect:.2f} seconds")
    print(f"Total processing time: {time_end - time_start:.2f} seconds")

    assert not df_aspects.empty, "Aspect extraction result is empty!"
    assert not df_sentiments.empty, "Sentiment classification result is empty!"
    