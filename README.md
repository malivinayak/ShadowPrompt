# ShadowPrompt: Confidential Query LLM

🔒 **Privacy-Preserving Retrieval-Augmented LLM System**
ShadowPrompt is a secure query platform that masks sensitive data (PII) before sending content to external LLMs. It ensures **privacy protection**, **fast retrieval**, and **accurate unmasking** of responses.

---

## ✨ Features

* **Automatic PII Detection & Masking**

  * Detects 25+ classes of PII (emails, phone numbers, SSNs, credit cards, etc.).
  * Replaces sensitive values with unique tokens (`<EMAIL>`, `<PHONE_NUMBER_1>`).

* **Privacy-Preserving Queries**

  * User text is **sanitized before hitting the LLM API**.
  * Final answers are unmasked locally to restore original terms.

* **Efficient RAG Pipeline**

  * Embedding-based retrieval using **BERT/BGE embeddings**.
  * Fast cosine-similarity search over stored documents.

* **Multi-Interface Support**

  * 🌐 **Flask Web App** (`flask_main.py`)
  * 📊 **Streamlit Dashboard** (`stream_main.py`)
  * 🧩 **Optimized v2 App** with heuristics + LangChain/Groq (`v2_main.py`).

* **Database Integration**

  * PostgreSQL backend for storing documents, embeddings, and PII mappings.
  * Automatic indexing for fast queries.

---

## 🏗️ Project Architecture

1. **Document Upload**

   * Detect PII → Mask text → Generate embedding → Store in DB.
2. **Query Processing**

   * Mask query PII → Embed → Find similar documents → LLM answer on **masked text only**.
3. **Response Unmasking**

   * Replace tokens with original values before returning answer.

---

## 🚀 Getting Started

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/shadowprompt.git
cd shadowprompt
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Key libraries:
`flask`, `streamlit`, `transformers`, `sentence-transformers`, `torch`,
`psycopg2-binary`, `numpy`, `pandas`, `scikit-learn`, `groq`, `python-dotenv`.

### 3. Setup Environment

Create a `.env` file:

```bash
DATABASE_PASSWORD=your_postgres_password
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key   # optional, for LangChain version
```

### 4. Setup PostgreSQL

```sql
CREATE DATABASE pii_database;
```

Tables are created automatically on first run.

### 5. Run Applications

* **Flask app**

  ```bash
  python flask_main.py
  # → http://localhost:8000
  ```
* **Streamlit app**

  ```bash
  streamlit run stream_main.py
  # → http://localhost:8501
  ```
* **Optimized v2 app**

  ```bash
  python v2_main.py
  # → http://localhost:8000
  ```

---

## 📂 Project Structure

```
shadowprompt/
│── flask_main.py      # Flask-based web app
│── stream_main.py     # Streamlit-based dashboard
│── v2_main.py         # Optimized v2 Flask app with heuristics
│── README.md          # Project documentation
```

---

## 📊 Example Workflow

1. Upload a `.txt` document containing sensitive info.
2. System detects and masks PII → stores masked version + mapping.
3. Query about the document (e.g., "What is John Doe’s account balance?").
4. Query is masked (`<NAME>`) → retrieved → sent to LLM.
5. Response is unmasked locally → **secure answer with real data restored**.

---

## 🔒 Why ShadowPrompt?

* **Privacy First** → Original PII never leaves your system.
* **Accuracy** → Masked/unmasked mapping preserves context.
* **Speed** → Embedding retrieval <0.8s latency.
* **Flexibility** → Supports multiple frontends (Flask, Streamlit).

---

## 📜 License

MIT License. Free to use and modify.

---
