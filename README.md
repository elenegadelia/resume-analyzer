# 📄 Resume / CV Analyzer with NLP

A professional NLP portfolio project that analyzes a candidate's resume against a job description using **TF-IDF vectorization**, **cosine similarity**, and **keyword gap analysis** - the same techniques used in real Applicant Tracking Systems (ATS).

> Built with Python · scikit-learn · spaCy · Streamlit · Plotly

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Resume Upload** | Supports PDF, DOCX, and plain text resumes |
| **Match Score** | Composite score: TF-IDF cosine similarity + keyword overlap |
| **Matched Keywords** | Terms from the job description already in your resume |
| **Missing Keywords** | High-priority terms from the job description not in your resume |
| **Top Resume Terms** | Most distinctive terms in your resume by TF-IDF weight |
| **NER Extraction** | Named entities (orgs, dates, locations) extracted with spaCy |
| **Visual Gauge** | Interactive score gauge with color-coded rating |
| **Bar Chart** | Top resume terms visualized with Plotly |
| **Improvement Tips** | Actionable suggestions for improving your resume |
| **Download Report** | Export the full analysis as a `.txt` file |

---

## 🧠 How It Works (NLP Concepts)

### TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF measures how important a word is to a document relative to a collection.
- **TF**: how often the word appears in this document
- **IDF**: how rare the word is across all documents
- Words like "Python" score high; words like "the" score near zero

### Cosine Similarity
Each document is represented as a vector (a list of TF-IDF weights). Cosine similarity measures the angle between two vectors:
- `1.0` (0° angle) = identical documents
- `0.0` (90° angle) = completely different documents

### Keyword Overlap Score
Extracts the top-weighted terms from the job description and checks what fraction appear in the resume. This directly mirrors how ATS systems work.

### Composite Score
```
score = (0.4 × TF-IDF cosine) + (0.6 × keyword overlap)
```
This blend is both technically grounded and intuitively meaningful.

### Why Not Transformers (BERT/GPT)?
TF-IDF is:
- ✅ Transparent and fully explainable
- ✅ Fast - runs in milliseconds on your laptop
- ✅ No API keys or internet required
- ✅ Standard interview topic for NLP/ML roles
- ✅ Directly mirrors real ATS systems

Transformers would add semantic understanding but are overkill for keyword-level resume matching. This project uses the right tool for the job.

---

## 🗂️ Project Structure

```
resume-analyzer/
├── app.py                  ← Streamlit UI (run this)
├── requirements.txt        ← Python dependencies
├── .gitignore
├── README.md
│
├── src/                    ← Core NLP logic (modular)
│   ├── __init__.py
│   ├── parser.py           ← Extract text from PDF/DOCX/TXT
│   ├── preprocessor.py     ← Clean and normalize text (spaCy)
│   ├── scorer.py           ← TF-IDF vectorization + cosine similarity
│   ├── keywords.py         ← Matched and missing keyword analysis
│   └── explainer.py        ← Plain-English result explanations
│
├── samples/                ← Sample files for testing
│   ├── sample_resume.txt
│   └── sample_job_description.txt
│
└── .streamlit/
    └── config.toml         ← UI theme configuration
```

---

## 🚀 Installation & Running Locally (macOS)

### Prerequisites
- Python 3.9+ (tested on 3.13)
- Terminal (zsh or bash)

### Step 1 — Clone the repository
```bash
git clone https://github.com/elenegadelia/resume-analyzer.git
cd resume-analyzer
```

### Step 2 — Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Download the spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🧪 Testing with Sample Files

Two sample files are included in `samples/`:

| File | Description |
|------|-------------|
| `sample_resume.txt` | A realistic data scientist / ML engineer resume |
| `sample_job_description.txt` | A machine learning engineer job posting |

To test:
1. Open the app (`streamlit run app.py`)
2. Upload `samples/sample_resume.txt`
3. Open `samples/sample_job_description.txt`, copy all text, paste into the job description field
4. Click **Analyze Resume**

---

## 📦 Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Core language |
| Streamlit | ≥1.43 | Web UI |
| scikit-learn | ≥1.6 | TF-IDF + cosine similarity |
| spaCy | ≥3.8 | Lemmatization + NER |
| PyMuPDF | ≥1.25 | PDF text extraction |
| python-docx | ≥1.1 | DOCX text extraction |
| pandas | ≥2.2 | Data manipulation |
| Plotly | ≥6.0 | Interactive visualizations |

---

## 🔮 Future Improvements

- [ ] BERT/sentence-transformer embeddings for semantic (not just keyword) matching
- [ ] Section detection (automatically find Skills, Experience, Education sections)
- [ ] Multi-resume comparison (rank N resumes against one job description)
- [ ] OCR support for scanned PDF resumes
- [ ] Database to track analysis history
- [ ] Export report as PDF

---

## 📌 Portfolio Notes

This project demonstrates:
- **Classical NLP fundamentals**: TF-IDF, cosine similarity, tokenization, stopword removal, lemmatization
- **Real-world application**: Mirrors how ATS systems actually work
- **Clean software architecture**: Modular `src/` package with separation of concerns
- **Production-quality UI**: Streamlit with custom CSS, Plotly charts, tab layout
- **File handling**: PDF, DOCX, and TXT parsing with error handling
- **Explainability**: Every score is explained in plain English

---

*Built as part of an AI Engineer portfolio.*
