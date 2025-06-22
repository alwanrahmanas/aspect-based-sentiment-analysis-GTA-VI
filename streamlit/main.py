import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("ABSA Sentiment Analysis 🚀")

texts = st.text_area("Masukkan review (pisahkan per baris):", height=200)

if st.button("Prediksi"):
    text_list = texts.strip().split("\n")
    response = requests.post(API_URL, json={"texts": text_list})

    if response.status_code == 200:
        results = response.json()["results"]   # ambil list-nya dulu
        for item in results:
            st.write(f"**Text**: {item['text']}")
            st.write("**Aspects & Sentiments:**")
            for aspect, sentiment in zip(item["aspects"], item["sentiments"]):
                st.write(f"- {aspect}: {sentiment}")
            st.markdown("---")

    else:
        st.error("Gagal prediksi, pastikan FastAPI sudah jalan.")
