import streamlit as st
from matcher import match_resume_with_jobs, extract_text_from_resume, generate_llm_insights
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =====================================================
# 🔹 PAGE CONFIGURATION
# =====================================================
st.set_page_config(page_title="AI Resume Matcher", page_icon="🤖", layout="centered")

st.title("🤖 AI Resume Matcher with LLM Insights")
st.write("Upload your resume and let AI instantly find the most relevant job matches!")

# =====================================================
# 🔹 FILE UPLOAD SECTION
# =====================================================
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.success("✅ Resume uploaded successfully!")

    # Save uploaded file locally
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # =====================================================
    # 🔹 MATCHING PROCESS
    # =====================================================
    with st.spinner("🔍 Matching your resume with job listings..."):
        results = match_resume_with_jobs(uploaded_file.name, top_k=5)
    st.success("✅ Matching completed successfully!")

    # =====================================================
    # 🔹 DISPLAY TOP JOB MATCHES
    # =====================================================
    st.subheader("🎯 Top Job Matches")
    for i, job in enumerate(results, 1):
        st.markdown(f"**{i}. {job['Job Title']}**")
        st.write(job["Job Description"])
        st.write(f"**Similarity Score:** {job['Similarity']:.2f}")
        if "Job Link" in job:
            st.markdown(f"[🔗 View Job Posting]({job['Job Link']})")
        st.divider()

    # =====================================================
    # 🔹 LLM INSIGHTS SECTION
    # =====================================================
    st.subheader("🧠 LLM-Powered Resume Insights")

    if st.button("✨ Generate Insights for Top Match"):
        top_job = results[0]
        resume_text = extract_text_from_resume(uploaded_file.name)

        with st.spinner("🧠 Generating AI insights... please wait..."):
            insight = generate_llm_insights(resume_text, top_job["Job Description"])

        # =====================================================
        # 🔹 BEAUTIFUL CHATGPT-STYLE DISPLAY
        # =====================================================
        # Detect Streamlit theme mode and adjust color dynamically
        # Detect Streamlit theme mode and adjust color dynamically
        dark_mode = st.get_option("theme.base") == "dark"

        bg_color = "#1E1E1E" if dark_mode else "#f7f9fc"
        text_color = "#F1F1F1" if dark_mode else "#000000"
        border_color = "#333333" if dark_mode else "#dfe3eb"

        st.markdown(f"""
        <div style="
             background-color:{bg_color};
             color:{text_color};
             border-radius:10px;
             padding:18px;
             border:1px solid {border_color};
             font-size:15px;
             line-height:1.6;
             white-space:pre-wrap;">
        {insight}
        </div>
        """, unsafe_allow_html=True)

        # =====================================================
        # 🔹 PDF DOWNLOAD FEATURE
        # =====================================================
        def generate_pdf_report(job_title, insight_text):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            styles = getSampleStyleSheet()

            elements = []
            elements.append(Paragraph("<b>AI Resume Matcher Report</b>", styles["Title"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"<b>Top Job Match:</b> {job_title}", styles["Normal"]))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("<b>AI Insights:</b>", styles["Heading2"]))
            elements.append(Paragraph(insight_text.replace("\n", "<br/>"), styles["Normal"]))
            elements.append(Spacer(1, 24))
            elements.append(Paragraph("Developed by <b>Chandan Kheto</b> 🚀", styles["Italic"]))

            doc.build(elements)
            buffer.seek(0)
            return buffer

        pdf_file = generate_pdf_report(top_job["Job Title"], insight)
        st.download_button(
            label="📥 Download AI Insights as PDF",
            data=pdf_file,
            file_name="AI_Resume_Insights.pdf",
            mime="application/pdf",
        )

# =====================================================
# 🔹 FOOTER
# =====================================================
st.markdown("---")
st.caption("Built with ❤️ by **Chandan Kheto** | Powered by 🧠 Mistral-7B & SentenceTransformers")
