<p align="center">
  <img src="https://i.imgur.com/Q7cAm9u.png" alt="ABSA GTA VI" width="250"/>
</p>

</p>

# 🎮 Aspect-Based Sentiment Analysis: GTA VI Edition 🚀

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi"/>
  <img src="https://img.shields.io/badge/Streamlit-1.34-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/HuggingFace-🤗-yellow"/>
</p>

This project is an implementation of Aspect-Based Sentiment Analysis (ABSA) for GTA VI reviews using a combination of:

- 🐍 **FastAPI** — Backend REST API
- 📊 **Streamlit** — Web UI interface
- 🎛️ **Gradio** (opsional) — Prototyping UI
- 🤖 **BERT Model** — Aspect extraction & sentiment classification

---

## 📂 Project's Structure
```
aspect-based-sentiment-analysis-GTA-VI/
├── app/
│   ├── api/
│   │   └── main.py                          # FastAPI app
│   ├── gradio/
│   │   └── main.py                          # Gradio interface (opsional)
│   └── inference/
│       ├── aspect_extraction/
│       │   ├── model.py                     # Aspect extraction model class
│       │   ├── main.py                      # Aspect extraction test model
│       │   └── label_order.json             # Aspect label order config
│       └── sentiment_classification/
│           ├── model.py                     # Sentiment classification model class
│           └── main.py                      # Sentiment classification test model
│
├── streamlit/
│   └── main.py                              # Streamlit app interface
│
├── test/
│   └── test_inference.py                    # Unit test for inference pipeline
│
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
├── render.yaml                              # Config FastAPI backend 
```
---

## 🚀 How to Run

### 📦 Install Dependency

```
pip install -r requirements.txt
```

🔥 Run FastAPI (Backend API)
```
uvicorn app.api.main:app --reload
```

📍 Access on: http://127.0.0.1:8000/docs

🎛️ Run Streamlit (Frontend)
```
streamlit run streamlit/main.py
```
📍 Access on: http://localhost:8501

(Optional) 🎚️ Run Gradio Interface
```
python app/gradio/main.py
```

📦 Feature API
POST /predict
Body:

```
{
  "texts": ["text review"]
}
Response:

{
  "result": [
    {
      "text": "text review",
      "aspects": ["Visual Graphic", "Nostalgia"],
      "sentiments": ["positive", "neutral"]
    }
  ]
}
```

## 📑 Author

**Alwan Rahmana S**

- 🚀 **GitHub:** [alwanrahmanas](https://github.com/alwanrahmanas)
- 📧 **Email:** [alwanrahmana@gmail.com](mailto:alwanrahmana@gmail.com)
- 💼 **LinkedIn:** [Alwan Rahmana](https://www.linkedin.com/in/alwanrahmana/)
- 🤗 **Hugging Face:** [alwanrahmana](https://huggingface.co/alwanrahmana/)

