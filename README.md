🧠 AI Resume Matcher with LLM Insights

An AI-powered Streamlit app that matches your resume to the most relevant job listings and generates smart career improvement insights using a Large Language Model (LLM).

🚀 Features

✅ AI-Powered Matching: Uses a transformer model (all-MiniLM-L6-v2) to find the best job matches for your resume.
✅ LLM Career Insights: Generates personalized feedback and improvement suggestions via Mistral-7B on Hugging Face.
✅ Multi-format Resume Support: Accepts .pdf, .docx, and .txt resumes.
✅ Interactive Streamlit UI: Simple, responsive web interface.
✅ PDF Export: Save AI insights as a PDF report using ReportLab.

🏗️ Project Structure
AI-Resume-Matcher/
│
├── app.py                   # Streamlit UI (main entry point)
├── matcher.py                # Core matching logic (embeddings + similarity)
├── preprocess.py             # Text preprocessing utilities
├── llm_assistant.py          # LLM insights generator using Hugging Face API
├── prepare_embeddings.py     # Precompute job embeddings for faster runtime
│
├── job_title_des.csv         # Job dataset with titles and descriptions
├── job_embeddings.pt         # Saved BERT embeddings for the job dataset
│
├── resume.docx               # Sample resume for testing
├── rsm.pdf                   # Sample generated report
│
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation

🧩 Tech Stack
Component	Technology Used
Frontend	Streamlit
NLP Model	SentenceTransformer (all-MiniLM-L6-v2)
LLM	Mistral 7B (via Hugging Face InferenceClient)
Data	CSV job dataset
Embedding Storage	Torch (.pt)
PDF Generation	ReportLab

⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/<your-username>/AI-Resume-Matcher.git
cd AI-Resume-Matcher

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Prepare job embeddings

Run this first to generate job_embeddings.pt from your dataset:

python prepare_embeddings.py

4️⃣ Add your Hugging Face token

Edit llm_assistant.py and set your token:

HF_TOKEN = "your_huggingface_token_here"

5️⃣ Launch the Streamlit app
streamlit run app.py

🧠 How It Works

1️⃣ User uploads a resume (PDF/DOCX/TXT).
2️⃣ Resume text is extracted and encoded into a dense vector using SentenceTransformer.
3️⃣ The app computes cosine similarity between the resume and precomputed job embeddings.
4️⃣ Top 5 most relevant jobs are displayed with similarity scores.
5️⃣ The top match is analyzed by an LLM (Mistral 7B) to generate personalized feedback and improvement suggestions.
6️⃣ The user can download the AI-generated report as a PDF.

🖼️ Screenshots
🧾 Upload Resume + Job Matches

🧠 AI Career Insights

📄 Downloadable PDF

(Replace above image links with your actual GitHub image paths after uploading screenshots.)

💡 Example Output

Similarity Results

1. Data Scientist — 0.89
2. Machine Learning Engineer — 0.86
3. AI Research Assistant — 0.83


LLM Insights Example

✅ Strengths: Strong Python and ML background.
⚠️ Missing Keywords: TensorFlow, MLOps.
💡 Suggestion: Add measurable impact metrics to your project section.

🧾 Future Improvements

🚀 Integrate LinkedIn Job Scraper to fetch real-time job listings.
🧠 Add multi-LLM comparison (OpenAI, Gemini, Mistral).
🎨 Improve UI with Tailwind + Streamlit Components.
🧩 Add resume scoring and visualization dashboard.

🤝 Contributing

Pull requests are welcome!
If you find a bug or want to add a feature, feel free to open an issue or submit a PR.

🧑‍💻 Author: Chandan Kheto
💼 AI/ML Engineer | NLP, LLMs, GenAI | Python, Hugging Face
📍 Based in India
