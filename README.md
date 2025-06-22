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

Project ini adalah implementasi **Aspect-Based Sentiment Analysis (ABSA)** untuk review GTA VI menggunakan kombinasi:

- 🐍 **FastAPI** — Backend REST API
- 📊 **Streamlit** — Web UI interface
- 🎛️ **Gradio** (opsional) — Prototyping UI
- 🤖 **BERT Model** — Aspect extraction & sentiment classification

---

## 📂 Struktur Project
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

```
---

## 🚀 Cara Menjalankan

### 📦 Install Dependency

```
pip install -r requirements.txt
```

🔥 Jalankan FastAPI (Backend API)
```
uvicorn app.api.main:app --reload
```

📍 Akses di: http://127.0.0.1:8000/docs

🎛️ Jalankan Streamlit (Frontend)
```
streamlit run streamlit/main.py
```
📍 Akses di: http://localhost:8501

(Opsional) 🎚️ Jalankan Gradio Interface
```
python app/gradio/main.py
```

📦 Fitur API
POST /predict
Body:

```
{
  "texts": ["text review pertama", "text review kedua"]
}
Response:

json
Copy
Edit
{
  "result": [
    {
      "text": "text review pertama",
      "aspects": ["story", "graphics"],
      "sentiments": ["positive", "neutral"]
    }
  ]
}
```

📑 Author
Alwan Rahmana

🚀 GitHub: alwanrahmanas
📧 alwanrahmana@gmail.com
