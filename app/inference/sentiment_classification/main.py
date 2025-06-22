from model import PredictSentiment
import pandas as pd

# ===== Contoh pemakaian =====
if __name__ == "__main__":
    # Contoh DataFrame input
    data = {
        'text': [
            "Very Bad trailer, the worst one in the history of game",
            "I would watch the trailer for another 100 years",
            "Such a nostalgic moment"
        ],
        'aspects': [
            ["Gameplay", "Trailer & Hype"],
            ["Gameplay", "Nostalgia"],
            ["Gameplay", "Nostalgia"]
        ],
        'proba_values': [
            [0.9782, 0.8123],
            [0.5985, 0.4789],
            [0.8624, 0.4191]
        ]
    }

    df_test = pd.DataFrame(data)

    sentiment_model = PredictSentiment()
    df_result = sentiment_model.predict(df_test)
    print(df_result)