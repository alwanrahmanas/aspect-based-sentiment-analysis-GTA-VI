import gradio as gr
from functools import lru_cache
from app.inference.aspect_extraction.model import PredictAspect
from app.inference.sentiment_classification.model import PredictSentiment

# Cache model agar load sekali saja
@lru_cache()
def get_aspect_model():
    return PredictAspect()

@lru_cache()
def get_sentiment_model():
    return PredictSentiment()

# Function prediksi end-to-end untuk Gradio
def predict_absa(text_input):
    texts = [text_input]

    # Load model dari cache
    aspect_model = get_aspect_model()
    sentiment_model = get_sentiment_model()

    # Prediksi
    df_aspects = aspect_model.predict(texts)
    df_sentiments = sentiment_model.predict(df_aspects)

    result = df_sentiments.to_dict(orient="records")

    result_str = ""
    for row in result:
        result_str += f"Text: {row['text']}\n"
        result_str += f"Aspect: {row['aspect']}\n"
        result_str += f"Sentiment: {row['predicted_sentiment']}\n\n"

    return result_str

# UI Gradio
iface = gr.Interface(
    fn=predict_absa,
    inputs=gr.Textbox(label="Enter your  comment", lines=4, placeholder="Thoughts on the game? Share your experience!"),
    outputs=gr.Textbox(label="Aspects and Sentiments", lines=10, placeholder="Results will be displayed here..."),
    title="<h1 style='color: #FFD700; text-align: center; font-family: \" impactful\", sans-serif; text-shadow: 2px 2px 4px #000000;'>Grand Theft Auto VI: Sentiment Showdown</h1>",
    description="<p style='color: #FFA500; text-align: center; font-family: \"impactful\", sans-serif;'>Drop your in-game thoughts and let's see if it's <b>wanted</b> or <b>wasted</b>!</p>",
    css="""
    body {
        background-image: url('https://upload.wikimedia.org/wikipedia/en/thumb/5/53/Grand_Theft_Auto_VI_logo.png/640px-Grand_Theft_Auto_VI_logo.png'); /* Replace with a suitable GTA VI background image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .gradio-container {
        background-color: rgba(0, 0, 0, 0.7); /* Darker, semi-transparent background for the main container */
        border: 3px solid #8B0000; /* Dark red border */
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(255, 0, 0, 0.7); /* Red glow effect */
    }
    .gr-textbox label {
        color: #00FF00 !important; /* Green text for labels */
        font-family: "impactful", sans-serif;
        font-size: 1.1em;
    }
    .gr-textbox textarea {
        background-color: #333333 !important; /* Darker input background */
        color: #FFD700 !important; /* Gold text in input */
        border: 1px solid #8B0000 !important;
        font-family: "monospace", monospace;
    }
    .gradio-button {
        background-color: #8B0000 !important; /* Dark red button */
        color: #FFD700 !important; /* Gold text on button */
        border: 2px solid #FFD700 !important;
        font-family: "impactful", sans-serif;
        font-size: 1.2em;
        text-shadow: 1px 1px 2px #000000;
    }
    .gradio-button:hover {
        background-color: #FF0000 !important; /* Brighter red on hover */
    }
    h1 {
        font-family: 'Impact', sans-serif; /* A font that resembles GTA titles */
        color: #FFD700; /* Gold color */
        text-shadow: 2px 2px 4px #000000; /* Black shadow */
    }
    p {
        font-family: 'Impact', sans-serif;
        color: #FFA500; /* Orange color */
    }
    """
)

# Run app
if __name__ == "__main__":
    iface.launch()