import streamlit as st
import requests

# === CONFIG ===
st.set_page_config(
    page_title="🎮 GTA VI Sentiment Analysis 🚀",
    page_icon="🔫", # Switched to a more iconic GTA icon
    layout="centered", # Changed back to centered for a more focused, elegant look
    initial_sidebar_state="collapsed"
)

API_URL = "http://127.0.0.1:8000/predict"
GITHUB_REPO_URL = "https://github.com/alwanrahmanas/aspect-based-sentiment-analysis-GTA-VI"

# Custom CSS for an elegant GTA VI theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bangers&family=Montserrat:wght@400;600;700&family=Roboto+Mono:wght@400&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background: #0D1117; /* Dark GitHub-like background */
        color: #E0E0E0; /* Light grey for general text */
        font-family: 'Montserrat', sans-serif;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0); /* Transparent header */
    }

    /* Main content container styling */
    .stApp {
        padding-top: 30px;
        padding-bottom: 30px;
        background-color: #161B22; /* Slightly lighter dark background for content area */
        border-radius: 12px;
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.1), 0 0 50px rgba(0, 255, 255, 0.05); /* Subtle cyan glow */
        border: 1px solid #2F363D; /* Subtle border */
    }

    h1 {
        font-family: 'Bangers', cursive; /* Main GTA-like font for the title */
        color: #FFD700 !important; /* Gold */
        text-shadow: 3px 3px 5px rgba(0,0,0,0.7), 0 0 10px rgba(255, 105, 180, 0.5); /* Black shadow + subtle hot pink glow */
        letter-spacing: 2px;
        font-size: 4.5rem !important; /* Slightly smaller for elegance */
        margin-bottom: 0px;
        line-height: 1.1;
    }

    h2 {
        font-family: 'Montserrat', sans-serif; /* Clean font for subheaders */
        color: #00FFC0 !important; /* Neon green/blue */
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        border-bottom: 1px solid #4A003D; /* Subtle dark purple line */
        padding-bottom: 8px;
        margin-top: 40px;
        font-weight: 600;
        font-size: 2rem !important;
    }

    h3 {
        font-family: 'Montserrat', sans-serif;
        color: #FF69B4 !important; /* Hot pink */
        font-size: 1.6rem !important;
        margin-top: 25px;
        font-weight: 700;
    }

    p {
        font-family: 'Montserrat', sans-serif;
        color: #C0C0C0;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Text Area Styling */
    .stTextArea label {
        color: #00FFC0 !important;
        font-family: 'Montserrat', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stTextArea textarea {
        background-color: #0D1117; /* Even darker input background */
        color: #FFD700; /* Gold text */
        border: 1px solid #4A003D; /* Dark purple border */
        border-radius: 8px;
        padding: 15px;
        font-family: 'Roboto Mono', monospace; /* Monospace for code-like input */
        font-size: 1.05rem;
        box-shadow: inset 0 0 8px rgba(74, 0, 61, 0.4); /* Inner glow */
    }

    /* Button Styling */
    .stButton > button {
        background-color: #DC143C; /* Crimson red */
        color: white !important;
        font-family: 'Montserrat', sans-serif; /* Clean font for button */
        font-size: 1.3rem;
        padding: 10px 25px;
        border-radius: 8px;
        border: 2px solid #FFD700; /* Gold border */
        box-shadow: 0 4px 12px rgba(220, 20, 60, 0.4); /* Red shadow */
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
    }
    .stButton > button:hover {
        background-color: #FF4500; /* OrangeRed on hover */
        box-shadow: 0 6px 15px rgba(255, 69, 0, 0.6);
        transform: translateY(-1px);
    }

    /* Spinner Styling */
    .stSpinner > div > div {
        border-top-color: #00FFC0 !important;
    }
    .stSpinner > div > div > div {
        color: #FF69B4 !important;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }

    /* Alert Styling */
    .stAlert {
        border-left: 5px solid #FFD700 !important;
        background-color: #2F363D !important; /* Darker grey */
        color: #FFD700 !important;
        font-family: 'Montserrat', sans-serif;
    }
    .stAlert > div > p {
        color: #FFD700 !important;
    }

    /* Success Message */
    .stSuccess {
        border-left: 5px solid #00FFC0 !important;
        background-color: #2F363D !important;
        color: #00FFC0 !important;
    }
    .stSuccess > div > p {
        color: #00FFC0 !important;
    }

    /* Horizontal Rules */
    hr {
        border-top: 1px dashed #4A003D; /* Subtle dashed purple line */
        margin-top: 35px;
        margin-bottom: 35px;
    }
    hr.dashed {
        border-top: 1px dashed #2F363D; /* Even lighter dashed line for internal separators */
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Result Display */
    .stExpander {
        border: 1px solid #2F363D;
        border-radius: 8px;
        margin-bottom: 15px;
        background-color: #1E222A; /* Slightly lighter background for expanders */
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .stExpander > div > div { /* For the expander header */
        background-color: #252A33; /* Darker header for expander */
        border-radius: 8px 8px 0 0;
        padding: 10px 15px;
        color: #FFD700;
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }
    .stExpander > div > div > div > p {
        color: #FFD700 !important; /* Color for the expander title */
        font-size: 1.1rem;
    }

    .stMarkdown b {
        color: #00FFC0 !important; /* Neon green/blue for bold text in results */
    }
    .stMarkdown code {
        background-color: #2A2A2A;
        color: #FFD700;
        border: 0.5px solid #4A003D;
        border-radius: 4px;
        padding: 2px 5px;
        font-family: 'Roboto Mono', monospace;
    }
    .sentiment-label {
        font-weight: bold;
        padding: 3px 8px;
        border-radius: 4px;
        display: inline-block;
        font-size: 0.95em;
        margin-left: 5px;
    }
    .sentiment-positive { background-color: rgba(0, 255, 192, 0.2); color: #00FFC0; border: 1px solid #00FFC0; }
    .sentiment-neutral { background-color: rgba(255, 215, 0, 0.2); color: #FFD700; border: 1px solid #FFD700; }
    .sentiment-negative { background-color: rgba(255, 105, 180, 0.2); color: #FF69B4; border: 1px solid #FF69B4; }


    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #4A003D; /* Subtle purple border for footer */
        color: #A0A0A0;
        font-size: 0.9em;
    }
    .footer a {
        color: #00FFC0; /* Neon green for links */
        text-decoration: none;
        font-weight: bold;
    }
    .footer a:hover {
        text-decoration: underline;
        color: #00FFFF; /* Brighter cyan on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === HEADER with IMAGE ===
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://i.imgur.com/Q7cAm9u.png" alt="ABSA GTA VI" width="300"/>
    </div>
    <h1 style='text-align: center;'>GTA VI: SENTIMENT SHOWDOWN</h1>
    <p style='text-align: center;'>
    Unleash the power of 🤖 BERT to dissect every whisper and roar from the streets of Vice City. 
    Our Aspect-Based Sentiment Analysis breaks down your GTA VI reviews, revealing the true feelings behind every detail.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# === INPUT AREA ===
st.subheader("📝 Drop Your Vice City Vibe Here")
texts = st.text_area(
    "Type each review on a new line. Tell us what you really think about the game!",
    height=200, # Slightly reduced height for elegance
    placeholder="Example:\nThe graphics are next-level, truly mind-blowing! 🤯\nThe new mission types feel fresh and keep me hooked for hours.\nDriving physics need some tweaks, feels a bit off sometimes."
)

# === BUTTON ===
if st.button("🚀 Analyze Sentiments, Partner!"):
    if texts.strip() == "":
        st.warning("💡 Hold up, cowboy! Please enter at least one review before running the analysis.")
    else:
        text_list = texts.strip().split("\n")
        # Filter out empty lines that might result from multiple newlines
        text_list = [t.strip() for t in text_list if t.strip()]

        with st.spinner("🔍 Cruising through your reviews... apprehending sentiments..."):
            try:
                response = requests.post(API_URL, json={"texts": text_list}, timeout=60) # Add timeout
                response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

                results = response.json()["results"] # Ensure this is 'result' based on your API

                st.success("✅ Analysis Complete! Vice City's verdict is in!")
                st.markdown("---")

                # === RESULT DISPLAY ===
                st.subheader("📊 Vice City's Sentiment Report")
                for idx, item in enumerate(results, start=1):
                    with st.expander(f"🎲 Review #{idx}: {item['text'][:70]}...", expanded=True if idx == 1 else False):
                        st.markdown(f"**📝 Original Text:** ```{item['text']}```") # Using code block for text

                        st.markdown("**🎯 Detected Aspects & Sentiments:**")
                        if item["aspects"]: # Ensure there are aspects before iterating
                            for aspect, sentiment in zip(item["aspects"], item["sentiments"]):
                                sentiment_emoji = ""
                                sentiment_class = ""
                                if sentiment == "positive":
                                    sentiment_emoji = "👍"
                                    sentiment_class = "sentiment-positive"
                                elif sentiment == "neutral":
                                    sentiment_emoji = "🤔"
                                    sentiment_class = "sentiment-neutral"
                                elif sentiment == "negative":
                                    sentiment_emoji = "👎"
                                    sentiment_class = "sentiment-negative"
                                
                                st.markdown(f"- **{aspect.capitalize()}**: <span class='sentiment-label {sentiment_class}'>{sentiment.capitalize()}</span> {sentiment_emoji}", unsafe_allow_html=True)
                        else:
                            st.info("No specific aspects detected for this review. Try providing more descriptive text!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Connection to FastAPI backend failed! Make sure it's running at `{API_URL}`. Error: {e}")
            except KeyError:
                 st.error(f"❌ Invalid response from API. Expected key 'result' but got: {response.json()}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# === FOOTER ===
st.markdown(
    f"""
    <div class="footer">
    <hr>
    <p>
    Developed with ❤️ in Los Santos by <strong>Alwan Rahmana</strong> 🚀<br>
    Explore the code on GitHub: <a href="{GITHUB_REPO_URL}" target="_blank">GTA VI Sentiment Showdown Repo</a><br>
    Powered by BERT, FastAPI, and Streamlit.
    </p>
    <p>
    📬 <strong>Contact:</strong><br>
    <a href="mailto:alwanrahmana@gmail.com">
        <img src="https://img.icons8.com/fluency/24/000000/apple-mail.png" alt="Email Icon" style="vertical-align: middle; margin-right: 5px;"/>
        alwanrahmana@gmail.com
    </a><br>
    <a href="https://www.linkedin.com/in/alwanrahmana/" target="_blank">
        <img src="https://img.icons8.com/color/24/000000/linkedin.png" alt="LinkedIn Icon" style="vertical-align: middle; margin-right: 5px;"/>
        LinkedIn
    </a><br>
    <a href="https://huggingface.co/alwanrahmana/" target="_blank">
        <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HF Icon" style="vertical-align: middle; margin-right: 5px; width:20px; height:20px;"/>
        Hugging Face
    </a>
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

