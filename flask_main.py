# Complete PII Masking and Retrieval System with Flask
# Requirements: pip install flask transformers sentence-transformers psycopg2-binary numpy pandas torch groq python-dotenv

from flask import Flask, render_template_string, request, jsonify, redirect, url_for
import psycopg2
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv
import re
from typing import Dict, List, Tuple, Any
import uuid
import time
from datetime import datetime

# Load environment variables
load_dotenv()

print(f"[{datetime.now()}] Starting PII Masking and Retrieval System...")

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'pii_database',
    'user': 'postgres',
    'password': os.getenv("DATABASE_PASSWORD")  # now pulled from .env or system env
}
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # now pulled from .env or system env

groq_client = Groq(api_key=GROQ_API_KEY)

print(f"[{datetime.now()}] Configuration loaded successfully")

# PII Classes
PII_CLASSES = [
    '<pin>', '<api_key>', '<bank_routing_number>', '<bban>', '<company>',
    '<credit_card_number>', '<credit_card_security_code>', '<customer_id>',
    '<date>', '<date_of_birth>', '<date_time>', '<driver_license_number>',
    '<email>', '<employee_id>', '<first_name>', '<iban>', '<ipv4>',
    '<ipv6>', '<last_name>', '<local_latlng>', '<name>', '<passport_number>',
    '<password>', '<phone_number>', '<social_security_number>',
    '<street_address>', '<swift_bic_code>', '<time>', '<user_name>'
]

print(f"[{datetime.now()}] PII Classes initialized: {len(PII_CLASSES)} classes loaded")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class DatabaseManager:
    def __init__(self):
        print(f"[{datetime.now()}] Initializing Database Manager...")
        self.conn = None
        self.connect()
        self.create_tables()
        print(f"[{datetime.now()}] Database Manager initialized successfully")
    
    def connect(self):
        print(f"[{datetime.now()}] Attempting to connect to PostgreSQL database...")
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            print(f"[{datetime.now()}] ✅ Successfully connected to PostgreSQL database")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Database connection failed: {e}")
            raise e
    
    def create_tables(self):
        print(f"[{datetime.now()}] Creating database tables if not exists...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) UNIQUE,
            original_text TEXT,
            masked_text TEXT,
            pii_mapping JSONB,
            embedding FLOAT8[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id);
        CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at);
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
                self.conn.commit()
            print(f"[{datetime.now()}] ✅ Database tables created/verified successfully")
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Table creation failed: {e}")
            raise e
    
    def store_document(self, doc_id: str, original_text: str, masked_text: str, 
                      pii_mapping: dict, embedding: np.ndarray):
        print(f"[{datetime.now()}] Storing document with ID: {doc_id}")
        print(f"[{datetime.now()}] Document length: {len(original_text)} characters")
        print(f"[{datetime.now()}] PII entities found: {len(pii_mapping)} items")
        print(f"[{datetime.now()}] Embedding dimensions: {embedding.shape}")
        
        insert_query = """
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
                cur.execute(insert_query, (
                    doc_id, original_text, masked_text, 
                    json.dumps(pii_mapping), embedding.tolist()
                ))
                self.conn.commit()
            print(f"[{datetime.now()}] ✅ Document stored successfully in database")
            return True
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Document storage failed: {e}")
            return False
    
    def get_similar_documents(self, query_embedding: np.ndarray, top_k: int = 5):
        print(f"[{datetime.now()}] Searching for similar documents (top_k={top_k})")
        print(f"[{datetime.now()}] Query embedding shape: {query_embedding.shape}")
        
        select_query = """
        SELECT doc_id, masked_text, pii_mapping, embedding, created_at
        FROM documents
        ORDER BY created_at DESC;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(select_query)
                results = cur.fetchall()
            
            print(f"[{datetime.now()}] Found {len(results)} documents in database")
            
            if not results:
                print(f"[{datetime.now()}] No documents found in database")
                return []
            
            # Calculate similarities
            print(f"[{datetime.now()}] Calculating cosine similarities...")
            similarities = []
            for i, (doc_id, masked_text, pii_mapping, embedding, created_at) in enumerate(results):
                if i % 10 == 0:  # Progress every 10 documents
                    print(f"[{datetime.now()}] Processing similarity for document {i+1}/{len(results)}")
                
                doc_embedding = np.array(embedding).reshape(1, -1)
                similarity = cosine_similarity(query_embedding.reshape(1, -1), doc_embedding)[0][0]
                similarities.append({
                    'doc_id': doc_id,
                    'masked_text': masked_text,
                    'pii_mapping': json.loads(pii_mapping),
                    'similarity': similarity,
                    'created_at': created_at
                })
            
            # Sort by similarity and return top_k
            print(f"[{datetime.now()}] Sorting documents by similarity...")
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]
            
            print(f"[{datetime.now()}] Top similarities:")
            for i, doc in enumerate(top_results):
                print(f"[{datetime.now()}]   {i+1}. Doc ID: {doc['doc_id'][:8]}... - Similarity: {doc['similarity']:.4f}")
            
            return top_results
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Similar document retrieval failed: {e}")
            return []

class PIIProcessor:
    def __init__(self):
        print(f"[{datetime.now()}] Initializing PII Processor...")
        self.pii_model = None
        self.pii_tokenizer = None
        self.embedding_model = None
        self.load_models()
        print(f"[{datetime.now()}] PII Processor initialized successfully")
    
    def load_models(self):
        print(f"[{datetime.now()}] Loading AI models...")
        try:
            print(f"[{datetime.now()}] Loading PII detection model...")
            self.pii_model = AutoModelForCausalLM.from_pretrained(
                "betterdataai/PII_DETECTION_MODEL"
            )
            self.pii_tokenizer = AutoTokenizer.from_pretrained(
                "betterdataai/PII_DETECTION_MODEL"
            )
            print(f"[{datetime.now()}] ✅ PII detection model loaded successfully")
            
            print(f"[{datetime.now()}] Loading sentence embedding model...")
            self.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", trust_remote_code=True)
            print(f"[{datetime.now()}] ✅ Sentence embedding model loaded successfully")
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Model loading failed: {e}")
            raise e
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        print(f"[{datetime.now()}] Starting PII detection...")
        print(f"[{datetime.now()}] Input text length: {len(text)} characters")
        print(f"[{datetime.now()}] Input text lines: {len(text.splitlines())} lines")
        print(f"[{datetime.now()}] First 200 chars: {repr(text[:200])}")
        
        if self.pii_model is None or self.pii_tokenizer is None:
            print(f"[{datetime.now()}] ❌ PII models not loaded")
            return {}
        
        # Preserve original text formatting and ensure multi-line handling
        formatted_text = text.strip()
        print(f"[{datetime.now()}] Formatted text length: {len(formatted_text)} characters")
        
        prompt = f"""You are an AI assistant for detecting Personal Identifiable Information (PII).

Your task is to identify and extract values from the given text based on the following 29 PII entity classes:

{', '.join(PII_CLASSES)}

Instructions:
1. Return the detected values grouped by class.
2. If no value exists for a class, skip that class.
3. Output should follow this format exactly:

<entity_class>: [value1, value2, ...]

Text:
\"\"\"{formatted_text}\"\"\"

Now extract and return the detected entities:

Output:
"""
        
        try:
            print(f"[{datetime.now()}] Tokenizing input for PII detection...")
            print(f"[{datetime.now()}] Prompt length: {len(prompt)} characters")
            
            # Use a larger max_length to accommodate longer texts
            inputs = self.pii_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            input_length = inputs['input_ids'].shape[1]
            print(f"[{datetime.now()}] Tokenized input length: {input_length} tokens")
            
            if input_length >= 8192:
                print(f"[{datetime.now()}] ⚠️ Input was truncated due to length limit")
            
            print(f"[{datetime.now()}] Generating PII detection response...")
            with torch.no_grad():
                outputs = self.pii_model.generate(
                    **inputs,
                    max_new_tokens=3000,  # Increased for longer outputs
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=self.pii_tokenizer.eos_token_id,
                    pad_token_id=self.pii_tokenizer.eos_token_id
                )
            
            print(f"[{datetime.now()}] Decoding PII detection output...")
            decoded = self.pii_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"[{datetime.now()}] Decoded output length: {len(decoded)} characters")
            
            # Extract only the output part
            if "Output:" in decoded:
                decoded = decoded.split("Output:")[-1].strip()
                print(f"[{datetime.now()}] Extracted output section length: {len(decoded)} characters")
            
            print(f"[{datetime.now()}] Raw PII detection output:")
            print(f"[{datetime.now()}] {repr(decoded[:500])}")  # Show first 500 chars
            
            print(f"[{datetime.now()}] Parsing PII entities from output...")
            pii_entities = self.parse_pii_output(decoded)
            
            print(f"[{datetime.now()}] ✅ PII detection completed")
            print(f"[{datetime.now()}] Found {len(pii_entities)} PII entity types:")
            for entity_type, values in pii_entities.items():
                print(f"[{datetime.now()}]   - {entity_type}: {len(values)} values - {values}")
            
            return pii_entities
            
        except Exception as e:
            print(f"[{datetime.now()}] ❌ PII detection failed: {e}")
            import traceback
            print(f"[{datetime.now()}] Full error traceback:")
            traceback.print_exc()
            return {}
    
    def parse_pii_output(self, output: str) -> Dict[str, List[str]]:
        print(f"[{datetime.now()}] Parsing PII output...")
        pii_entities = {}
        lines = output.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if ':' in line and any(cls in line for cls in PII_CLASSES):
                try:
                    entity_class, values_str = line.split(':', 1)
                    entity_class = entity_class.strip()
                    
                    # Parse the values
                    values_str = values_str.strip()
                    if values_str.startswith('[') and values_str.endswith(']'):
                        values_str = values_str[1:-1]
                    
                    values = [v.strip().strip('"\'') for v in values_str.split(',') if v.strip()]
                    
                    if values and values != ['']:
                        pii_entities[entity_class] = values
                        if i % 5 == 0:  # Progress every 5 entities
                            print(f"[{datetime.now()}] Parsed entity {i+1}: {entity_class} with {len(values)} values")
                except:
                    continue
        
        print(f"[{datetime.now()}] ✅ PII parsing completed: {len(pii_entities)} entity types")
        return pii_entities
    
    def create_masked_text_and_mapping(self, text: str, pii_entities: Dict[str, List[str]]) -> Tuple[str, Dict[str, str]]:
        print(f"[{datetime.now()}] Creating masked text and PII mapping...")
        print(f"[{datetime.now()}] Original text length: {len(text)} characters")
        print(f"[{datetime.now()}] Original text lines: {len(text.splitlines())} lines")
        print(f"[{datetime.now()}] Original text preview: {repr(text[:200])}")
        
        masked_text = text  # Preserve original text structure
        pii_mapping = {}
        total_replacements = 0
        
        # Create unique tokens for each PII value
        for entity_class, values in pii_entities.items():
            print(f"[{datetime.now()}] Processing {entity_class} with {len(values)} values: {values}")
            for i, value in enumerate(values):
                if value and len(value.strip()) > 0:
                    # Create unique token
                    if len(values) == 1:
                        token = entity_class
                    else:
                        token = f"{entity_class}_{i+1}"
                    
                    # Count occurrences before replacement
                    original_count = masked_text.count(value)
                    
                    # Replace in text (case-sensitive exact match)
                    if original_count > 0:
                        masked_text = masked_text.replace(value, token)
                        pii_mapping[token] = value
                        total_replacements += original_count
                        
                        print(f"[{datetime.now()}] Replaced '{value}' with '{token}' ({original_count} occurrences)")
                    else:
                        print(f"[{datetime.now()}] ⚠️ Value '{value}' not found in text for replacement")
        
        print(f"[{datetime.now()}] ✅ Masking completed: {total_replacements} total replacements made")
        print(f"[{datetime.now()}] Final mapping contains {len(pii_mapping)} unique tokens")
        print(f"[{datetime.now()}] Masked text length: {len(masked_text)} characters")
        print(f"[{datetime.now()}] Masked text lines: {len(masked_text.splitlines())} lines")
        print(f"[{datetime.now()}] Masked text preview: {repr(masked_text[:200])}")
        
        # Verify text integrity
        if len(masked_text) == 0:
            print(f"[{datetime.now()}] ❌ WARNING: Masked text is empty!")
            return text, {}  # Return original text if masking failed
        
        return masked_text, pii_mapping
    
    def generate_embedding(self, text: str) -> np.ndarray:
        print(f"[{datetime.now()}] Generating text embedding...")
        print(f"[{datetime.now()}] Text length for embedding: {len(text)} characters")
        
        if self.embedding_model is None:
            print(f"[{datetime.now()}] ❌ Embedding model not loaded")
            return np.array([])
        
        try:
            start_time = time.time()
            embedding = self.embedding_model.encode(
                [text], 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            end_time = time.time()
            
            print(f"[{datetime.now()}] ✅ Embedding generated in {end_time - start_time:.2f} seconds")
            print(f"[{datetime.now()}] Embedding shape: {embedding[0].shape}")
            return embedding[0]
        except Exception as e:
            print(f"[{datetime.now()}] ❌ Embedding generation failed: {e}")
            return np.array([])

class LLMProcessor:
    @staticmethod
    def query_llm(query: str, context_docs: List[Dict], combined_mapping: Dict[str, str]) -> str:
        print(f"[{datetime.now()}] Querying LLM with {len(context_docs)} context documents...")
        print(f"[{datetime.now()}] Query: {query[:100]}..." if len(query) > 100 else f"[{datetime.now()}] Query: {query}")
        
        # Prepare context from similar documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['masked_text']}"
            for i, doc in enumerate(context_docs)
        ])
        
        print(f"[{datetime.now()}] Context length: {len(context)} characters")
        
        prompt = f"""
You are an AI assistant that answers questions based on the provided context documents. 
The documents contain masked PII data using tokens like <name>, <email>, etc.

Context Documents:
{context}

Query: {query}

Please provide a comprehensive answer based on the context documents. Use the masked tokens as they appear in the documents.

Answer:
"""
        
        try:
            print(f"[{datetime.now()}] Sending request to Groq API...")
            start_time = time.time()
            
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            end_time = time.time()
            print(f"[{datetime.now()}] ✅ LLM response received in {end_time - start_time:.2f} seconds")
            
            response_text = response.choices[0].message.content
            print(f"[{datetime.now()}] Response length: {len(response_text)} characters")
            
            return response_text
        except Exception as e:
            print(f"[{datetime.now()}] ❌ LLM query failed: {e}")
            return "Sorry, I couldn't process your request due to an error."
    
    @staticmethod
    def unmask_response(response: str, combined_mapping: Dict[str, str]) -> str:
        print(f"[{datetime.now()}] Unmasking response...")
        print(f"[{datetime.now()}] Available tokens for unmasking: {len(combined_mapping)}")
        
        unmasked_response = response
        replacements_made = 0
        
        # Replace masked tokens with original values
        for token, original_value in combined_mapping.items():
            if token in unmasked_response:
                count = unmasked_response.count(token)
                unmasked_response = unmasked_response.replace(token, original_value)
                replacements_made += count
                print(f"[{datetime.now()}] Replaced '{token}' with '{original_value}' ({count} times)")
        
        print(f"[{datetime.now()}] ✅ Unmasking completed: {replacements_made} total replacements made")
        
        return unmasked_response

# Global instances
print(f"[{datetime.now()}] Initializing global instances...")
db_manager = DatabaseManager()
pii_processor = PIIProcessor()
print(f"[{datetime.now()}] ✅ All global instances initialized successfully")

# HTML Templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PII Masking & Retrieval System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .nav { text-align: center; margin-bottom: 30px; }
        .nav a { display: inline-block; padding: 12px 25px; margin: 0 10px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background-color 0.3s; }
        .nav a:hover { background: #0056b3; }
        .nav a.active { background: #28a745; }
        .form-group { margin-bottom: 20px; }
        label { display: block; font-weight: bold; margin-bottom: 5px; color: #555; }
        input[type="file"], textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        textarea { resize: vertical; min-height: 100px; }
        button { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background-color 0.3s; }
        button:hover { background: #0056b3; }
        .alert { padding: 15px; margin: 20px 0; border-radius: 5px; }
        .alert-success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .alert-error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .alert-info { background: #d1ecf1; border: 1px solid #bee5eb; color: #0c5460; }
        .results { margin-top: 30px; }
        .result-section { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #007bff; }
        .similarity-score { font-weight: bold; color: #007bff; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
        .progress { margin: 20px 0; padding: 15px; background: #e9ecef; border-radius: 5px; font-family: monospace; }
        .two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        @media (max-width: 768px) { .two-column { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔒 PII Masking & Retrieval System</h1>
        
        <div class="nav">
            <a href="/" class="{{ 'active' if mode == 'upload' else '' }}">📤 Document Upload</a>
            <a href="/query" class="{{ 'active' if mode == 'query' else '' }}">🔍 Query Documents</a>
        </div>
        
        {% if mode == 'upload' %}
        <h2>Document Upload & Processing</h2>
        <form method="POST" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Upload a text file:</label>
                <input type="file" id="file" name="file" accept=".txt" required>
            </div>
            <button type="submit">Process Document</button>
        </form>
        
        {% elif mode == 'query' %}
        <h2>Query Processing & Retrieval</h2>
        <form method="POST">
            <div class="form-group">
                <label for="query">Enter your query:</label>
                <textarea id="query" name="query" placeholder="Ask a question about your uploaded documents..." required>{{ query or '' }}</textarea>
            </div>
            <button type="submit">Process Query</button>
        </form>
        {% endif %}
        
        {% if messages %}
        <div class="results">
            {% for message in messages %}
            <div class="alert alert-{{ message.type }}">{{ message.content }}</div>
            {% endfor %}
        </div>
        {% endif %}
        
        {% if results %}
        <div class="results">
            {{ results | safe }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def upload_page():
    print(f"[{datetime.now()}] User accessed upload page")
    return render_template_string(HTML_TEMPLATE, mode='upload')

@app.route('/', methods=['POST'])
def upload_document():
    print(f"[{datetime.now()}] Document upload request received")
    
    if 'file' not in request.files:
        print(f"[{datetime.now()}] ❌ No file provided in request")
        return render_template_string(HTML_TEMPLATE, mode='upload', messages=[{'type': 'error', 'content': 'No file uploaded'}])
    
    file = request.files['file']
    if file.filename == '':
        print(f"[{datetime.now()}] ❌ Empty filename provided")
        return render_template_string(HTML_TEMPLATE, mode='upload', messages=[{'type': 'error', 'content': 'No file selected'}])
    
    try:
        print(f"[{datetime.now()}] Processing uploaded file: {file.filename}")
        text_content = file.read().decode('utf-8')
        print(f"[{datetime.now()}] File read successfully, length: {len(text_content)} characters")
        
        # Step 1: Detect PII
        print(f"[{datetime.now()}] === STEP 1: DETECTING PII ENTITIES ===")
        pii_entities = pii_processor.detect_pii(text_content)
        
        if not pii_entities:
            print(f"[{datetime.now()}] ⚠️ No PII entities detected in document")
            return render_template_string(HTML_TEMPLATE, mode='upload', 
                                        messages=[{'type': 'info', 'content': 'No PII entities detected in the document'}])
        
        print(f"[{datetime.now()}] === STEP 2: CREATING MASKED TEXT AND MAPPING ===")
        # Step 2: Create masked text and mapping
        masked_text, pii_mapping = pii_processor.create_masked_text_and_mapping(text_content, pii_entities)
        
        print(f"[{datetime.now()}] === STEP 3: GENERATING EMBEDDINGS ===")
        # Step 3: Generate embedding
        embedding = pii_processor.generate_embedding(text_content)
        
        if embedding.size == 0:
            print(f"[{datetime.now()}] ❌ Failed to generate embeddings")
            return render_template_string(HTML_TEMPLATE, mode='upload', 
                                        messages=[{'type': 'error', 'content': 'Failed to generate embeddings'}])
        
        print(f"[{datetime.now()}] === STEP 4: STORING IN DATABASE ===")
        # Step 4: Store in database
        doc_id = str(uuid.uuid4())
        success = db_manager.store_document(doc_id, text_content, masked_text, pii_mapping, embedding)
        
        if success:
            print(f"[{datetime.now()}] ✅ DOCUMENT PROCESSING COMPLETED SUCCESSFULLY!")
            
            results_html = f"""
            <div class="result-section">
                <h3>✅ Document Processed Successfully!</h3>
                <p><strong>Document ID:</strong> {doc_id}</p>
                <p><strong>Original Length:</strong> {len(text_content)} characters</p>
                <p><strong>PII Entities Found:</strong> {len(pii_mapping)} items</p>
                <p><strong>Entity Types:</strong> {', '.join(pii_entities.keys())}</p>
            </div>
            
            <div class="two-column">
                <div>
                    <h4>Masked Text</h4>
                    <pre>{masked_text[:500]}{'...' if len(masked_text) > 500 else ''}</pre>
                </div>
                <div>
                    <h4>PII Mapping</h4>
                    <pre>{json.dumps(pii_mapping, indent=2)}</pre>
                </div>
            </div>
            """
            
            return render_template_string(HTML_TEMPLATE, mode='upload', 
                                        messages=[{'type': 'success', 'content': 'Document processed and stored successfully!'}],
                                        results=results_html)
        else:
            print(f"[{datetime.now()}] ❌ Failed to store document in database")
            return render_template_string(HTML_TEMPLATE, mode='upload', 
                                        messages=[{'type': 'error', 'content': 'Failed to store document in database'}])
    
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Document processing failed: {str(e)}")
        return render_template_string(HTML_TEMPLATE, mode='upload', 
                                    messages=[{'type': 'error', 'content': f'Document processing failed: {str(e)}'}])

@app.route('/query')
def query_page():
    print(f"[{datetime.now()}] User accessed query page")
    return render_template_string(HTML_TEMPLATE, mode='query')

@app.route('/query', methods=['POST'])
def process_query():
    print(f"[{datetime.now()}] Query processing request received")
    
    query = request.form.get('query', '').strip()
    if not query:
        print(f"[{datetime.now()}] ❌ Empty query provided")
        return render_template_string(HTML_TEMPLATE, mode='query', 
                                    messages=[{'type': 'error', 'content': 'Please enter a query'}])
    
    try:
        print(f"[{datetime.now()}] Processing query: {query[:100]}..." if len(query) > 100 else f"[{datetime.now()}] Processing query: {query}")
        
        print(f"[{datetime.now()}] === STEP 1: PROCESSING QUERY FOR PII ===")
        # Step 1: Process query for PII
        query_pii = pii_processor.detect_pii(query)
        query_masked, query_mapping = pii_processor.create_masked_text_and_mapping(query, query_pii)
        
        print(f"[{datetime.now()}] === STEP 2: GENERATING QUERY EMBEDDING ===")
        # Step 2: Generate query embedding
        query_embedding = pii_processor.generate_embedding(query)
        
        if query_embedding.size == 0:
            print(f"[{datetime.now()}] ❌ Failed to generate query embedding")
            return render_template_string(HTML_TEMPLATE, mode='query', query=query,
                                        messages=[{'type': 'error', 'content': 'Failed to generate query embedding'}])
        
        print(f"[{datetime.now()}] === STEP 3: FINDING SIMILAR DOCUMENTS ===")
        # Step 3: Find similar documents
        similar_docs = db_manager.get_similar_documents(query_embedding, top_k=5)
        
        if not similar_docs:
            print(f"[{datetime.now()}] ⚠️ No similar documents found")
            return render_template_string(HTML_TEMPLATE, mode='query', query=query,
                                        messages=[{'type': 'info', 'content': 'No similar documents found in the database'}])
        
        print(f"[{datetime.now()}] === STEP 4: COMBINING PII MAPPINGS ===")
        # Step 4: Combine mappings
        combined_mapping = query_mapping.copy()
        for doc in similar_docs:
            combined_mapping.update(doc['pii_mapping'])
        
        print(f"[{datetime.now()}] Combined mapping contains {len(combined_mapping)} unique tokens")
        
        print(f"[{datetime.now()}] === STEP 5: QUERYING LLM ===")
        # Step 5: Query LLM
        llm_response = LLMProcessor.query_llm(query_masked, similar_docs, combined_mapping)
        
        print(f"[{datetime.now()}] === STEP 6: UNMASKING RESPONSE ===")
        # Step 6: Unmask response
        final_response = LLMProcessor.unmask_response(llm_response, combined_mapping)
        
        print(f"[{datetime.now()}] ✅ QUERY PROCESSING COMPLETED SUCCESSFULLY!")
        
        # Generate results HTML
        similar_docs_html = ""
        for i, doc in enumerate(similar_docs):
            similar_docs_html += f"""
            <div class="result-section">
                <h4>Document {i+1} <span class="similarity-score">(Similarity: {doc['similarity']:.4f})</span></h4>
                <p><strong>Document ID:</strong> {doc['doc_id']}</p>
                <p><strong>Created:</strong> {doc['created_at']}</p>
                <p><strong>Content Preview:</strong></p>
                <pre>{doc['masked_text'][:300]}{'...' if len(doc['masked_text']) > 300 else ''}</pre>
            </div>
            """
        
        results_html = f"""
        <div class="result-section">
            <h3>📊 Query Analysis</h3>
            <p><strong>Original Query:</strong> {query}</p>
            <p><strong>Masked Query:</strong> {query_masked}</p>
            {f'<p><strong>Query PII Mapping:</strong></p><pre>{json.dumps(query_mapping, indent=2)}</pre>' if query_mapping else ''}
        </div>
        
        <div class="result-section">
            <h3>📄 Similar Documents ({len(similar_docs)} found)</h3>
            {similar_docs_html}
        </div>
        
        <div class="result-section">
            <h3>🤖 Final Response</h3>
            <div style="background: #e8f5e8; padding: 15px; border-radius: 5px; border-left: 4px solid #28a745;">
                {final_response.replace('\n', '<br>')}
            </div>
        </div>
        """
        
        return render_template_string(HTML_TEMPLATE, mode='query', query=query,
                                    messages=[{'type': 'success', 'content': f'Query processed successfully! Found {len(similar_docs)} relevant documents.'}],
                                    results=results_html)
    
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Query processing failed: {str(e)}")
        return render_template_string(HTML_TEMPLATE, mode='query', query=query,
                                    messages=[{'type': 'error', 'content': f'Query processing failed: {str(e)}'}])

@app.route('/status')
def status():
    """Health check endpoint"""
    print(f"[{datetime.now()}] Status check requested")
    try:
        # Check database connection
        with db_manager.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            doc_count = cur.fetchone()[0]
        
        status_info = {
            'status': 'healthy',
            'database_connected': True,
            'documents_stored': doc_count,
            'models_loaded': {
                'pii_model': pii_processor.pii_model is not None,
                'pii_tokenizer': pii_processor.pii_tokenizer is not None,
                'embedding_model': pii_processor.embedding_model is not None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[{datetime.now()}] Status check completed: {doc_count} documents in database")
        return jsonify(status_info)
    
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Status check failed: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/docs')
def list_documents():
    """List all stored documents"""
    print(f"[{datetime.now()}] Document list requested")
    try:
        with db_manager.conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, 
                       LEFT(original_text, 100) as preview,
                       JSONB_OBJECT_KEYS(pii_mapping) as pii_types,
                       created_at 
                FROM documents 
                ORDER BY created_at DESC;
            """)
            docs = cur.fetchall()
        
        docs_html = "<h2>📚 Stored Documents</h2>"
        
        if not docs:
            docs_html += "<p>No documents found in the database.</p>"
        else:
            for doc_id, preview, pii_types, created_at in docs:
                docs_html += f"""
                <div class="result-section">
                    <h4>Document ID: {doc_id}</h4>
                    <p><strong>Created:</strong> {created_at}</p>
                    <p><strong>Preview:</strong> {preview}...</p>
                </div>
                """
        
        print(f"[{datetime.now()}] Document list generated: {len(docs)} documents")
        return render_template_string(HTML_TEMPLATE, mode='docs', results=docs_html)
    
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Document listing failed: {e}")
        return render_template_string(HTML_TEMPLATE, mode='docs', 
                                    messages=[{'type': 'error', 'content': f'Failed to list documents: {str(e)}'}])

if __name__ == '__main__':
    print(f"[{datetime.now()}] Starting Flask application...")
    print(f"[{datetime.now()}] Server will be available at: http://localhost:5000")
    print(f"[{datetime.now()}] Available endpoints:")
    print(f"[{datetime.now()}]   - GET  /        : Document upload page")
    print(f"[{datetime.now()}]   - POST /        : Process uploaded document")
    print(f"[{datetime.now()}]   - GET  /query   : Query interface")
    print(f"[{datetime.now()}]   - POST /query   : Process query")
    print(f"[{datetime.now()}]   - GET  /status  : System status (JSON)")
    print(f"[{datetime.now()}]   - GET  /docs    : List stored documents")
    print(f"[{datetime.now()}] ========================================")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=8000, threaded=True)