from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from functools import lru_cache

from app.inference.aspect_extraction.model import PredictAspect
from app.inference.sentiment_classification.model import PredictSentiment

app = FastAPI()

# Request payload schema
class SentimentRequest(BaseModel):
    texts: list

# Dependency Injection with cache
@lru_cache()
def get_aspect_model():
    return PredictAspect()

@lru_cache()
def get_sentiment_model():
    return PredictSentiment()

# Root test
@app.get("/")
def read_root():
    return {"status": "running"}

# Endpoint prediksi
@app.post("/predict")
def predict_sentiment(
    request: SentimentRequest,
    aspect_model: PredictAspect = Depends(get_aspect_model),
    sentiment_model: PredictSentiment = Depends(get_sentiment_model)
):
    texts = request.texts
    if not texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty.")

    # Aspect Extraction
    df_aspects = aspect_model.predict(texts)
    # Sentiment Classification
    df_sentiments = sentiment_model.predict(df_aspects)

    # Organize result per text
    result_list = []
    for text in texts:
        df_filtered = df_sentiments[df_sentiments["text"] == text]
        aspects = df_filtered["aspect"].tolist()
        sentiments = df_filtered["predicted_sentiment"].tolist()

        result_list.append({
            "text": text,
            "aspects": aspects,
            "sentiments": sentiments
        })

    return {"results": result_list}
# Run the app with: uvicorn app.api.main:app --reload
# Access the API at: http://