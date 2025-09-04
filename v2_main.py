# app.py
import os
import json
import re
import time
import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv

# DB
import psycopg2
from psycopg2.extras import Json, register_default_json, register_default_jsonb

# ML / Embeddings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LLM: Groq client
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# LangChain optional (used for orchestration if available)
try:
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# Load env
load_dotenv()

# ---------------------------
# Configuration (tweak me)
# ---------------------------
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_NAME', 'pii_database'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DATABASE_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))
}

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if GROQ_API_KEY and GROQ_AVAILABLE:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None

# Embedding model name (SentenceTransformer)
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL', "BAAI/bge-large-en-v1.5")

# PII classes (your existing list)
PII_CLASSES = [
    '<pin>', '<api_key>', '<bank_routing_number>', '<bban>', '<company>',
    '<credit_card_number>', '<credit_card_security_code>', '<customer_id>',
    '<date>', '<date_of_birth>', '<date_time>', '<driver_license_number>',
    '<email>', '<employee_id>', '<first_name>', '<iban>', '<ipv4>',
    '<ipv6>', '<last_name>', '<local_latlng>', '<name>', '<passport_number>',
    '<password>', '<phone_number>', '<social_security_number>',
    '<street_address>', '<swift_bic_code>', '<time>', '<user_name>'
]

# ---------------------------
# Flask app
# ---------------------------
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'change-me-in-prod')

# ---------------------------
# Utilities
# ---------------------------
def now():
    return datetime.now().isoformat(sep=' ', timespec='seconds')

def coerce_json(value):
    """
    Robust converter: accepts dict, JSON string, bytes, memoryview, None.
    Returns python object (dict) or {} on failure.
    """
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, memoryview):
        try:
            value = bytes(value).decode("utf-8")
        except Exception:
            return {}
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8")
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            # maybe it's a Python dict string? try eval fallback (safe-ish because data comes from DB)
            try:
                return json.loads(value.replace("'", '"'))
            except Exception:
                return {}
    return {}

# ---------------------------
# Database manager
# ---------------------------
class DatabaseManager:
    def __init__(self, cfg: dict):
        print(f"[{now()}] DB: initializing")
        self.cfg = cfg
        self.conn = None
        self.connect()
        self.create_tables()
        print(f"[{now()}] DB: ready")

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.cfg)
            # ensure psycopg2 returns JSONB as Python native types
            register_default_json(loads=json.loads, globally=True)
            register_default_jsonb(loads=json.loads, globally=True)
            print(f"[{now()}] DB: connected")
        except Exception as e:
            print(f"[{now()}] DB: connect error: {e}")
            raise e

    def create_tables(self):
        q = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) UNIQUE,
            original_text TEXT,
            masked_text TEXT,
            pii_mapping JSONB,
            embedding FLOAT8[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_docid ON documents(doc_id);
        """
        with self.conn.cursor() as cur:
            cur.execute(q)
            self.conn.commit()

    def store_document(self, doc_id: str, original_text: str, masked_text: str, pii_mapping: dict, embedding: np.ndarray):
        print(f"[{now()}] DB: storing doc {doc_id} (entities={len(pii_mapping)})")
        q = """
        INSERT INTO documents (doc_id, original_text, masked_text, pii_mapping, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (doc_id) DO UPDATE SET
            original_text = EXCLUDED.original_text,
            masked_text = EXCLUDED.masked_text,
            pii_mapping = EXCLUDED.pii_mapping,
            embedding = EXCLUDED.embedding;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(q, (doc_id, original_text, masked_text, Json(pii_mapping), embedding.tolist()))
                self.conn.commit()
            return True
        except Exception as e:
            print(f"[{now()}] DB: store error: {e}")
            return False

    def list_all_documents(self):
        q = "SELECT doc_id, masked_text, pii_mapping, embedding, created_at FROM documents ORDER BY created_at DESC;"
        with self.conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
        # rows: list of tuples
        docs = []
        for doc_id, masked_text, pii_mapping, embedding, created_at in rows:
            docs.append({
                "doc_id": doc_id,
                "masked_text": masked_text,
                "pii_mapping": coerce_json(pii_mapping),
                "embedding": np.array(embedding) if embedding is not None else None,
                "created_at": created_at
            })
        return docs

# ---------------------------
# Embedding + PII processor
# ---------------------------
class PIIProcessor:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL_NAME):
        print(f"[{now()}] Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name, device='cuda' if torch.cuda.is_available() else 'cpu', trust_remote_code=True)
        # PII detection LLM (your model). We keep existing method where you call a model; here we assume model is loaded similarly.
        # To keep compatibility we preserve your pipelined approach (user previously used AutoModelForCausalLM).
        # If you have a specialized fine-tuned T5 available, plug it here and a wrapper for mask inference.
        self.pii_tokenizer = None
        self.pii_model = None
        # NOTE: user already had "betterdataai/PII_DETECTION_MODEL" — we keep dynamic loading on demand to avoid heavy startup time.
        try:
            # Try to lazily load only if env asks
            model_name = os.getenv("PII_MODEL", None)
            if model_name:
                print(f"[{now()}] Loading PII detection model: {model_name}")
                self.pii_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.pii_model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"[{now()}] PII detection model loaded")
        except Exception as e:
            print(f"[{now()}] Warning: failed to load PII detection model at startup: {e}")
            self.pii_tokenizer = None
            self.pii_model = None

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """
        If you have a fine-tuned PII model, call it here and return a dict: {ENTITY_TOKEN: [values...]}
        For safety, if model missing, return no entities.
        """
        if not self.pii_model or not self.pii_tokenizer:
            # Fallback: very small heuristic extraction for common tokens (email, phone, date)
            return self._heuristic_extract(text)
        # If model present, run and parse output (your original code can slot in)
        prompt = f"""You are an assistant that extracts PII classes... (use same prompt as before)
Text:
\"\"\"{text}\"\"\"
Output:
"""
        inputs = self.pii_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to(self.pii_model.device)
        with torch.no_grad():
            out = self.pii_model.generate(**inputs, max_new_tokens=1024)
        decoded = self.pii_tokenizer.decode(out[0], skip_special_tokens=True)
        # parse using previous parse rules
        return self.parse_pii_output(decoded)

    def _heuristic_extract(self, text: str) -> Dict[str, List[str]]:
        # Lightweight heuristics — keeps system working without your fine-tuned model
        entities = {}
        # emails
        emails = re.findall(r'[\w\.-]+@[\w\.-]+', text)
        if emails:
            entities['<email>'] = list(dict.fromkeys(emails))
        # phones (simple)
        phones = re.findall(r'\b(?:\+?\d{1,3}[-\s.]*)?(?:\d{3}[-\s.]*){1,4}\d{3,4}\b', text)
        phones = [p for p in phones if len(re.sub(r'\D','',p)) >= 7]
        if phones:
            entities['<phone_number>'] = list(dict.fromkeys(phones))
        # dates (very simple)
        dates = re.findall(r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b', text)
        if dates:
            entities['<date>'] = list(dict.fromkeys(dates))
        # names heuristic (very naive: capitalized words)
        names = re.findall(r'\b[A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}\b', text)
        # exclude some false positives like "If You"
        names = [n for n in names if len(n.split()) <= 3]
        if names:
            entities['<name>'] = list(dict.fromkeys(names))
        return entities

    def parse_pii_output(self, output: str) -> Dict[str, List[str]]:
        # Simple parser: read lines like "<entity>: [val1, val2]"
        res = {}
        lines = output.strip().splitlines()
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            left, right = line.split(':', 1)
            left = left.strip()
            right = right.strip()
            if not right:
                continue
            # normalize bracket list
            if right.startswith('[') and right.endswith(']'):
                inner = right[1:-1]
                items = [x.strip().strip('"\'') for x in inner.split(',') if x.strip()]
            else:
                items = [right]
            if items:
                res[left] = items
        return res

    def create_masked_text_and_mapping(self, text: str, pii_entities: Dict[str, List[str]]) -> Tuple[str, Dict[str, str]]:
        masked = text
        mapping = {}
        for entity_class, vals in pii_entities.items():
            for i, v in enumerate(vals):
                if not v or not v.strip():
                    continue
                # Normalize token: remove angle brackets and use uppercase label, add index if multiple
                base = entity_class.strip().strip('<>').upper()
                token = f"<{base}>" if len(vals) == 1 else f"<{base}_{i+1}>"
                # Replace exact occurrences - case-sensitive, but also replace common variants
                # Use re.escape for safe replacement
                try:
                    # Count occurrences
                    count = len(re.findall(re.escape(v), masked))
                except Exception:
                    count = 0
                if count > 0:
                    masked = masked.replace(v, token)
                    mapping[token] = v
                else:
                    # try case-insensitive replace
                    pattern = re.compile(re.escape(v), flags=re.IGNORECASE)
                    if pattern.search(masked):
                        masked = pattern.sub(token, masked)
                        mapping[token] = v
        return masked, mapping

    def generate_embedding(self, text: str) -> np.ndarray:
        if text is None or len(text.strip()) == 0:
            return np.array([])
        emb = self.embedding_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return np.array(emb[0], dtype=np.float32)

# ---------------------------
# Similarity index (in-memory)
# ---------------------------
class SimilarityIndex:
    """
    Simple in-memory index: load embeddings from DB on startup or on demand.
    For large corpora, replace with FAISS / Milvus / Pinecone.
    """
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.docs = []  # list of dicts with keys doc_id, masked_text, pii_mapping, embedding, created_at
        self.emb_matrix = None  # NxD numpy array
        self._load_from_db()

    def _load_from_db(self):
        self.docs = self.db.list_all_documents()
        embeddings = []
        for d in self.docs:
            if d.get('embedding') is None:
                embeddings.append(np.zeros(1, dtype=np.float32))  # placeholder
            else:
                embeddings.append(np.array(d['embedding'], dtype=np.float32))
        if embeddings:
            # stack -> matrix
            try:
                self.emb_matrix = np.vstack([e.reshape(1, -1) for e in embeddings])
            except Exception:
                # fallback: make empty
                self.emb_matrix = None
        else:
            self.emb_matrix = None
        print(f"[{now()}] Index loaded: {len(self.docs)} docs")

    def refresh(self):
        self._load_from_db()

    def search(self, query_emb: np.ndarray, top_k: int = 5):
        if self.emb_matrix is None or len(self.docs) == 0:
            return []
        # ensure shapes
        q = query_emb.reshape(1, -1)
        # cosine similarity
        sims = cosine_similarity(q, self.emb_matrix)[0]  # shape (N,)
        idxs = np.argsort(sims)[::-1][:top_k]
        results = []
        for i in idxs:
            d = self.docs[int(i)]
            results.append({
                "doc_id": d["doc_id"],
                "masked_text": d["masked_text"],
                "pii_mapping": d["pii_mapping"],
                "similarity": float(sims[int(i)]),
                "created_at": d["created_at"]
            })
        return results

# ---------------------------
# LLM orchestration (LangChain optional)
# ---------------------------
def llm_answer(masked_query: str, context_docs: List[Dict], combined_mapping: Dict[str, str]) -> str:
    """
    Use LangChain ChatOpenAI if installed (and OPENAI_API_KEY present) else fallback to Groq client if available.
    We always pass masked docs; we MUST NOT send original PII to the LLM.
    """
    # Build context string
    context = "\n\n".join([f"Document {i+1}:\n{d['masked_text']}" for i, d in enumerate(context_docs)])
    prompt = f"""You are a helpful assistant. Use only the provided context to answer.
All PII in the context is masked with tokens like <NAME>, <PHONE_NUMBER>, <DATE_1>, etc.
Context:
{context}

Question (masked): {masked_query}

Answer concisely, preserving masked tokens as-is.
"""
    # If langchain available and OPENAI_API_KEY present, use ChatOpenAI
    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(temperature=0.0)
            msgs = [SystemMessage(content="You are a helpful assistant that answers using provided context."), HumanMessage(content=prompt)]
            resp = llm(messages=msgs)
            return resp.content
        except Exception as e:
            print(f"[{now()}] LangChain LLM error: {e}")
    # Fallback: Groq
    if groq_client:
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers using provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[{now()}] Groq error: {e}")
    # If no LLM available
    return "No LLM provider available (set OPENAI_API_KEY for LangChain/OpenAI or GROQ_API_KEY for Groq)."

# ---------------------------
# Unmasking utility
# ---------------------------
def unmask_text(response: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return response
    # replace tokens in descending length order to avoid partial overlaps
    # Accept forms: <TOKEN>, [TOKEN], TOKEN
    keys = sorted(mapping.keys(), key=len, reverse=True)
    out = response
    for k in keys:
        v = mapping[k]
        k_clean = k.strip()
        # forms
        variants = [k_clean, k_clean.strip('<>'), k_clean.strip('[]')]
        for var in variants:
            # replace <VAR> and [VAR] and VAR (word-boundary)
            out = out.replace(var, v)
            out = out.replace(f"<{var}>", v)
            out = out.replace(f"[{var}]", v)
            # word boundary
            out = re.sub(rf"\b{re.escape(var)}\b", v, out)
    return out

# ---------------------------
# HTML (same as you had, simplified for brevity)
# ---------------------------
HTML_TEMPLATE = """..."""  # (You can reuse your existing template string to keep UI identical)

# For brevity in this snippet, I'll reuse a simple minimal HTML template
HTML_TEMPLATE = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>PII RAG</title>
<style>body{font-family:Arial;padding:20px;}textarea{width:100%;height:120px}</style>
</head>
<body>
<h1>PII Masking & Retrieval</h1>
<nav><a href="/">Upload</a> | <a href="/query">Query</a></nav>
<hr/>
{% if mode == 'upload' %}
<form method="post" enctype="multipart/form-data">
<input type="file" name="file" accept=".txt"/><br/><br/>
<button type="submit">Upload</button>
</form>
{% elif mode == 'query' %}
<form method="post">
<textarea name="query" required>{{ query or '' }}</textarea><br/><br/>
<button type="submit">Search</button>
</form>
{% endif %}
<hr/>
{% if messages %}
<ul>{% for m in messages %}<li>{{ m }}</li>{% endfor %}</ul>{% endif %}
{% if results %}{{ results | safe }}{% endif %}
</body>
</html>
"""

# ---------------------------
# Global instances
# ---------------------------
print(f"[{now()}] Starting app - loading components...")
db_manager = DatabaseManager(DB_CONFIG)
pii_processor = PIIProcessor()
indexer = SimilarityIndex(db_manager)
print(f"[{now()}] Components initialized.")

# ---------------------------
# Routes
# ---------------------------
@app.route('/', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, mode='upload', messages=None, results=None)
    # POST - file upload
    file = request.files.get('file')
    if not file:
        return render_template_string(HTML_TEMPLATE, mode='upload', messages=["No file provided"], results=None)
    try:
        raw = file.read().decode('utf-8')
        # Detect PII
        pii_entities = pii_processor.detect_pii(raw)
        masked_text, mapping = pii_processor.create_masked_text_and_mapping(raw, pii_entities)
        # IMPORTANT: embed masked_text (not original)
        embedding = pii_processor.generate_embedding(masked_text)
        if embedding.size == 0:
            return render_template_string(HTML_TEMPLATE, mode='upload', messages=["Failed to create embedding"], results=None)
        doc_id = str(uuid.uuid4())
        ok = db_manager.store_document(doc_id, raw, masked_text, mapping, embedding)
        # refresh index
        indexer.refresh()
        if ok:
            results_html = f"<h3>Stored doc {doc_id}</h3><pre>{masked_text[:500]}</pre><pre>{json.dumps(mapping, indent=2)}</pre>"
            return render_template_string(HTML_TEMPLATE, mode='upload', messages=["Document processed successfully"], results=results_html)
        else:
            return render_template_string(HTML_TEMPLATE, mode='upload', messages=["DB error storing document"], results=None)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, mode='upload', messages=[f"Processing error: {e}"], results=None)

@app.route('/query', methods=['GET', 'POST'])
def query_page():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE, mode='query', messages=None, results=None, query=None)
    # POST - process query
    query = request.form.get('query', '').strip()
    if not query:
        return render_template_string(HTML_TEMPLATE, mode='query', messages=["Empty query"], results=None, query=query)
    try:
        # Detect PII in query and mask it
        q_pii = pii_processor.detect_pii(query)
        q_masked, q_map = pii_processor.create_masked_text_and_mapping(query, q_pii)
        # Generate embedding of masked query (IMPORTANT)
        q_emb = pii_processor.generate_embedding(q_masked)
        if q_emb.size == 0:
            return render_template_string(HTML_TEMPLATE, mode='query', messages=["Failed to embed query"], results=None, query=query)
        # Search index
        candidates = indexer.search(q_emb, top_k=5)
        if not candidates:
            return render_template_string(HTML_TEMPLATE, mode='query', messages=["No similar documents found"], results=None, query=query)
        # Merge mappings (query mapping first so query tokens preferred)
        combined_map = {}
        combined_map.update(q_map)
        for c in candidates:
            if isinstance(c.get('pii_mapping'), dict):
                combined_map.update(c['pii_mapping'])
        # Ask LLM using masked query + masked docs
        llm_resp = llm_answer(q_masked, candidates, combined_map)
        final = unmask_text(llm_resp, combined_map)
        # Build results html
        docs_html = ""
        for i, d in enumerate(candidates):
            docs_html += f"<h4>Doc {i+1} (sim={d['similarity']:.4f})</h4><pre>{d['masked_text'][:400]}</pre>"
        resp_html = f"<h3>Answer</h3><div><pre>{final}</pre></div><hr/>{docs_html}"
        return render_template_string(HTML_TEMPLATE, mode='query', messages=["Query processed"], results=resp_html, query=query)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, mode='query', messages=[f"Query failed: {e}"], results=None, query=query)

@app.route('/status')
def status():
    try:
        docs = db_manager.list_all_documents()
        info = {
            "status": "healthy",
            "documents": len(docs),
            "models_loaded": {
                "embedding_model": pii_processor.embedding_model is not None,
                "pii_model": pii_processor.pii_model is not None
            },
            "timestamp": now()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

# ---------------------------
# Run app
# ---------------------------
if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    print(f"[{now()}] Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
