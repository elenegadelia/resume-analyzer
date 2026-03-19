"""
app.py — Streamlit UI for the Resume / CV Analyzer.

This is the entry point of the application.
Run with: streamlit run app.py

Architecture:
  - This file handles the UI and user interaction only.
  - All NLP logic lives in src/ modules (parser, preprocessor, scorer, keywords, explainer).
  - This keeps the app clean: UI code and business logic are separated.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.parser import extract_text_from_bytes
from src.preprocessor import preprocess, light_clean, get_named_entities
from src.scorer import compute_similarity, get_top_tfidf_terms
from src.keywords import analyze_keywords, get_resume_highlights
from src.explainer import (
    generate_explanation,
    generate_improvement_suggestions,
    score_label,
    score_color,
)

# ─── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume Analyzer | NLP Portfolio",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 700;
            color: #1f2937;
            margin-bottom: 0;
        }
        .sub-header {
            font-size: 1rem;
            color: #6b7280;
            margin-top: 0;
        }
        .score-box {
            border-radius: 12px;
            padding: 24px;
            text-align: center;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #374151;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .keyword-chip {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            margin: 3px;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">📄 Resume / CV Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">NLP-powered resume analysis using TF-IDF and cosine similarity</p>',
    unsafe_allow_html=True,
)
st.divider()

# ─── Input Section ─────────────────────────────────────────────────────────────

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.subheader("1. Upload Your Resume")
    st.caption("Supported formats: PDF, DOCX, TXT")
    uploaded_file = st.file_uploader(
        "Choose a resume file",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

with col_right:
    st.subheader("2. Paste Job Description")
    st.caption("Copy the full job posting text here")
    job_description = st.text_area(
        "Job description",
        height=220,
        placeholder="Paste the full job description here...\n\nInclude requirements, responsibilities, and qualifications for best results.",
        label_visibility="collapsed",
    )

st.divider()

# ─── Analysis Button ───────────────────────────────────────────────────────────

analyze_clicked = st.button("🔍 Analyze Resume", type="primary", use_container_width=True)

# ─── Main Analysis Logic ───────────────────────────────────────────────────────

if analyze_clicked:
    # Input validation
    if not uploaded_file:
        st.error("Please upload a resume file.")
        st.stop()
    if not job_description.strip():
        st.error("Please paste a job description.")
        st.stop()
    if len(job_description.strip()) < 100:
        st.warning("The job description seems very short. For best results, paste the full posting.")

    # ── Extract Resume Text ───────────────────────────────────────────────────
    with st.spinner("Extracting text from resume..."):
        try:
            resume_raw = extract_text_from_bytes(
                uploaded_file.read(), uploaded_file.name
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if len(resume_raw.strip()) < 50:
        st.error(
            "Could not extract enough text from the resume. "
            "If it's a scanned PDF (image), text extraction won't work — "
            "please use a digital/text-based PDF or copy-paste as TXT."
        )
        st.stop()

    # ── Preprocess Both Documents ─────────────────────────────────────────────
    with st.spinner("Preprocessing text..."):
        # light_clean: used for TF-IDF scoring (preserves phrases like "machine learning")
        resume_for_tfidf = light_clean(resume_raw)
        job_for_tfidf = light_clean(job_description)
        # preprocess: full lemmatization for NER and token analysis
        resume_clean = preprocess(resume_raw)

    # ── Compute Similarity Score ──────────────────────────────────────────────
    with st.spinner("Computing similarity score..."):
        results = compute_similarity(resume_for_tfidf, job_for_tfidf)
        score = results["score"]
        vectorizer = results["vectorizer"]
        resume_vector = results["resume_vector"]
        job_vector = results["job_vector"]
        feature_names = results["feature_names"]

    # ── Keyword Analysis ──────────────────────────────────────────────────────
    with st.spinner("Analyzing keywords..."):
        keyword_results = analyze_keywords(
            resume_for_tfidf, job_vector, feature_names, top_n=30
        )
        matched = keyword_results["matched"]
        missing = keyword_results["missing"]

        resume_highlights = get_resume_highlights(resume_vector, feature_names, top_n=15)

    # ── Named Entity Recognition ──────────────────────────────────────────────
    with st.spinner("Extracting entities from resume..."):
        entities = get_named_entities(resume_raw)

    # ─── Results Display ──────────────────────────────────────────────────────

    st.divider()
    st.subheader("Analysis Results")

    # Score + Gauge Chart
    color = score_color(score)
    label = score_label(score)

    res_col1, res_col2 = st.columns([1, 2], gap="large")

    with res_col1:
        # Show component scores for educational transparency
        tfidf_score = results.get("tfidf_score", 0)
        keyword_overlap = results.get("keyword_overlap", 0)
        st.caption(f"TF-IDF Cosine: {tfidf_score:.1f}% · Keyword Overlap: {keyword_overlap:.1f}%")

        # Gauge chart using Plotly
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": label, "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 20], "color": "#fee2e2"},
                        {"range": [20, 40], "color": "#fed7aa"},
                        {"range": [40, 60], "color": "#fef9c3"},
                        {"range": [60, 80], "color": "#bbf7d0"},
                        {"range": [80, 100], "color": "#86efac"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": score,
                    },
                },
            )
        )
        fig.update_layout(height=280, margin=dict(t=30, b=0, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Score stats
        st.metric("Matched Keywords", len(matched))
        st.metric("Missing Keywords", len(missing))

    with res_col2:
        # Plain-English Explanation
        st.markdown("### What This Score Means")
        explanation = generate_explanation(score, matched, missing)
        st.markdown(explanation)

    st.divider()

    # Keyword Detail Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["✅ Matched Keywords", "❌ Missing Keywords", "📊 Top Resume Terms", "🔍 Extracted Info"]
    )

    with tab1:
        st.markdown("**Keywords from the job description found in your resume:**")
        if matched:
            # Display as colored chips
            chips_html = " ".join(
                f'<span style="background:#dcfce7; color:#166534; padding:4px 12px; '
                f'border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem;">'
                f'✓ {kw}</span>'
                for kw in matched
            )
            st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.info("No strong keyword matches found. Consider rewriting your resume to mirror the job description.")

    with tab2:
        st.markdown("**Important job description terms missing from your resume:**")
        if missing:
            chips_html = " ".join(
                f'<span style="background:#fee2e2; color:#991b1b; padding:4px 12px; '
                f'border-radius:20px; margin:3px; display:inline-block; font-size:0.85rem;">'
                f'✗ {kw}</span>'
                for kw in missing
            )
            st.markdown(chips_html, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Improvement Suggestions:**")
            suggestions = generate_improvement_suggestions(missing, score)
            for s in suggestions:
                st.markdown(f"• {s}")
        else:
            st.success("Your resume covers all major keywords from the job description!")

    with tab3:
        st.markdown("**The most distinctive terms in your resume (by TF-IDF weight):**")
        if resume_highlights:
            df = pd.DataFrame(resume_highlights, columns=["Term", "TF-IDF Weight"])

            # Horizontal bar chart
            fig2 = go.Figure(
                go.Bar(
                    x=df["TF-IDF Weight"],
                    y=df["Term"],
                    orientation="h",
                    marker_color="#6366f1",
                )
            )
            fig2.update_layout(
                height=400,
                xaxis_title="TF-IDF Weight",
                yaxis={"categoryorder": "total ascending"},
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No terms could be extracted.")

    with tab4:
        st.markdown("**Entities extracted from your resume using spaCy NER:**")
        st.caption(
            "NER (Named Entity Recognition) automatically identifies real-world entities "
            "like organizations, dates, and locations in text."
        )

        # Show a curated subset of useful entity types
        useful_labels = {
            "ORG": "Organizations / Companies",
            "GPE": "Locations",
            "DATE": "Dates",
            "PERSON": "Names",
            "PRODUCT": "Products / Tools",
            "EVENT": "Events",
            "WORK_OF_ART": "Projects / Works",
            "LAW": "Certifications / Standards",
        }

        found_any = False
        for label, display_name in useful_labels.items():
            if label in entities and entities[label]:
                st.markdown(f"**{display_name}:**")
                st.markdown(", ".join(f"`{v}`" for v in entities[label][:10]))
                found_any = True

        if not found_any:
            st.info("No notable entities were detected. This is normal for plain-text resumes with minimal formatting.")

        # Also show raw resume text for reference
        with st.expander("View extracted resume text"):
            st.text(resume_raw[:3000] + ("..." if len(resume_raw) > 3000 else ""))

    # ── Download Report ────────────────────────────────────────────────────────

    st.divider()
    st.markdown("### Download Report")

    report_lines = [
        "RESUME ANALYZER — ANALYSIS REPORT",
        "=" * 50,
        f"Match Score: {score:.1f}% — {label}",
        "",
        "MATCHED KEYWORDS:",
        ", ".join(matched) if matched else "None",
        "",
        "MISSING KEYWORDS:",
        ", ".join(missing) if missing else "None",
        "",
        "EXPLANATION:",
        generate_explanation(score, matched, missing).replace("**", "").replace("*", ""),
        "",
        "IMPROVEMENT SUGGESTIONS:",
    ]
    for s in generate_improvement_suggestions(missing, score):
        report_lines.append(f"• {s}")

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Analysis Report (.txt)",
        data=report_text,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
    )

# ─── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    "<center><small>Built with Python · scikit-learn · spaCy · Streamlit · Plotly</small></center>",
    unsafe_allow_html=True,
)
