import torch
import pandas as pd
import streamlit as st
import torch.serialization as serialization
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import docx
import os
import requests
from huggingface_hub import InferenceClient

# =====================================================
# 🔹 CONFIGURATION
# =====================================================
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = "job_embeddings.pt"  # must match file from prepare_embeddings.py

# Initialize Hugging Face LLM client
HF_TOKEN = os.getenv("HF_TOKEN")  # read token from environment
client = InferenceClient(model = "mistralai/Mistral-7B-Instruct-v0.3", token=HF_TOKEN)

# =====================================================
# 🔹 LOAD MODEL + PRECOMPUTED EMBEDDINGS
# =====================================================
print("🚀 Loading model and precomputed embeddings...")
model = SentenceTransformer(MODEL_NAME)

if not os.path.exists(EMBEDDINGS_PATH):
    raise FileNotFoundError("❌ job_embeddings.pt not found! Run prepare_embeddings.py first.")

serialization.add_safe_globals([pd.DataFrame])  # allow pandas DataFrame loading

data = torch.load(EMBEDDINGS_PATH, weights_only=False)
job_embeddings = data["embeddings"]
job_data = data["df"]

print(f"✅ Loaded {len(job_data)} job entries from dataset.")

# =====================================================
# 🔹 FUNCTION: EXTRACT TEXT FROM RESUME
# =====================================================
def extract_text_from_resume(file_path):
    text = ""
    if file_path.endswith(".pdf"):
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
    return text.strip()

# =====================================================
# 🔹 FUNCTION: MATCH RESUME WITH JOBS
# =====================================================
def match_resume_with_jobs(resume_path, top_k=5):
    # extract resume text & embed
    resume_text = extract_text_from_resume(resume_path)
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)

    # compute cosine similarities (tensor)
    similarities = util.cos_sim(resume_embedding, job_embeddings)[0]

    # topk returns tensors (values, indices)
    top_results = torch.topk(similarities, k=top_k)

    # Convert indices and scores into plain Python lists of ints/floats
    # Use .cpu().numpy() to handle GPU/CPU tensors, then convert to Python types
    top_values = top_results.values.detach().cpu().numpy().tolist()
    top_indices = top_results.indices.detach().cpu().numpy().astype(int).tolist()

    results = []
    for score, idx in zip(top_values, top_indices):
        # idx is now a plain integer safe to use with .iloc
        row = job_data.iloc[idx]

        job_info = {
            "Job Title": row.get("Job Title", "N/A"),
            "Job Description": (row.get("Job Description", "")[:200] + "...") if row.get("Job Description", "") else "",
            "Similarity": float(score)  # make sure it's a python float
        }

        if "Job Link" in job_data.columns:
            job_info["Job Link"] = row.get("Job Link", "")

        results.append(job_info)

    return results

# =====================================================
# 🔹 FUNCTION: LLM-Powered Resume Insights
# =====================================================
def generate_llm_insights(resume_text, top_jobs):
    try:
        prompt = f"""
You are an AI career coach.
Below is a candidate's resume and top 5 matched job descriptions.
Explain:
1️⃣ Why these jobs match or don't match.
2️⃣ What important skills or keywords are missing in the resume.
3️⃣ Give 2-3 improvement tips for better matching in the future.

Resume:
{resume_text[:1500]}

Top Jobs:
{top_jobs}
"""

        messages = [
            {"role": "system", "content": "You are a helpful AI career advisor."},
            {"role": "user", "content": prompt},
        ]

        # ✅ Use chat_completion() instead of conversational()
        response = client.chat_completion(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            messages=messages,
            max_tokens=500,
        )

        return response.choices[0].message["content"]

    except Exception as e:
        return f"⚠️ Error generating insights: {str(e)}"
