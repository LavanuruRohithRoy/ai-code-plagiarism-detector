# 🚀 AI‑Assisted Code Plagiarism Detection System

> 🔍 *A similarity‑based plagiarism detection engine for source code — **not** an AI authorship classifier.*

---

## ✨ Overview

This project implements an **end‑to‑end code plagiarism detection system** that focuses on **semantic, structural, and token‑level similarity** rather than surface‑level text matching.

Instead of asking *“Was this written by AI?”*, the system answers the more realistic and industry‑relevant question:

> **“How similar is this code to existing code?”**

This mirrors how real plagiarism engines work in academic and enterprise environments.

---

## 🧠 What This System Does

* ✅ Accepts source code as input
* ✅ Normalizes and parses code
* ✅ Performs **AST‑based structural analysis**
* ✅ Computes **token‑level similarity**
* ✅ Generates **semantic embeddings** using pretrained transformers
* ✅ Performs fast similarity search using **FAISS**
* ✅ Produces plagiarism scores and confidence metrics

---

## 🏗️ System Architecture

```
User Code
   ↓
FastAPI Layer
   ↓
Code Normalizer
   ↓
AST Analyzer
   ↓
Token Similarity Engine
   ↓
Embedding Model (CodeBERT)
   ↓
FAISS Similarity Search
   ↓
Score Aggregator
   ↓
Results + Storage
```

Each layer is isolated to ensure **modularity**, **debuggability**, and **scalability**.

---

## 🛠️ Tech Stack

* 🐍 **Python**
* ⚡ **FastAPI**
* 🔥 **PyTorch**
* 🤗 **HuggingFace Transformers (CodeBERT)**
* 🌳 **Python AST**
* 🚀 **FAISS** (Vector Similarity Search)
* 🗄️ **SQLite** (for production usage)
* 📊 **Pandas & Matplotlib** (evaluation & visualization)

---

## 📂 Project Structure

```
ai-code-plagiarism-detector/
│
├── src/
│   ├── api/            # FastAPI routes
│   ├── pipeline/       # Core analysis pipeline
│   ├── storage/        # DB & FAISS logic
│   ├── utils/          # Shared helpers (future use)
│   └── configs/        # Config placeholders
│
├── scripts/
│   ├── evaluate_dataset.py
│   ├── analyze_results.py
│   └── plot_results.py
│
├── data/
│   ├── raw/            # Human & AI datasets
│   └── results/        # CSV outputs
│
└── README.md
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone <https://github.com/LavanuruRohithRoy/ai-code-plagiarism-detector.git>
cd ai-code-plagiarism-detector
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Dataset & Evaluation

### 📌 Dataset Used

* **Human Code:** Competitive programming solutions (LeetCode‑style)
* **AI Code:** Python programs generated using large language models

> ⚠️ Evaluation is performed **offline** to avoid polluting production data.

---

## 📊 Evaluation Results & Visualizations

### 📦 Plagiarism Percentage Distribution

![Plagiarism Distribution](assets/plagiarism_boxplot.png)

**What this shows:**

* Human code exhibits higher similarity due to repeated algorithmic patterns
* AI code remains clustered near lower similarity values
* High plagiarism ≠ misconduct — it indicates reuse

---

### 🤖 AI Probability Distribution

![AI Probability Distribution](assets/ai_probability_boxplot.png)

**What this shows:**

* AI probability reflects **similarity confidence**, not authorship
* Human solutions show higher variance due to standardized styles

---

### 📈 Plagiarism Percentage Histogram

![Plagiarism Histogram](assets/plagiarism_histogram.png)

**What this shows:**

* Human code forms a dense similarity cluster
* AI samples concentrate near zero similarity
* Behavior matches real‑world plagiarism engines

---

## 📌 Key Learnings

* Difference between **similarity‑based systems** and **classifiers**
* Why dataset imbalance does **not** bias similarity engines
* Designing production‑ready ML pipelines
* Separating evaluation logic from live systems
* Debugging real ML infra issues (DB, FAISS, embeddings)

---

## 🔮 Future Work

✨ Add explainability / reasoning for plagiarism scores
✨ Improve API‑level result interpretation
✨ Optimize performance for large‑scale datasets
✨ Extend support to more programming languages

---

## ⚠️ Disclaimer

This system measures **code similarity and reuse**, not authorship verification. High plagiarism scores indicate similarity patterns and do **not** imply unethical behavior.

---

🚧 **Project Status:** Actively under development

