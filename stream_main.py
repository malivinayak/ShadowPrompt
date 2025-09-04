# Complete PII Masking and Retrieval System
# Requirements: pip install streamlit transformers sentence-transformers psycopg2-binary numpy pandas torch groq python-dotenv

import streamlit as st
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

# Load environment variables
load_dotenv()

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'pii_database',
    'user': 'postgres',
    'password': os.getenv("DATABASE_PASSWORD")
}

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_client = Groq(api_key=GROQ_API_KEY)

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

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            st.success("Connected to PostgreSQL database")
        except Exception as e:
            st.error(f"Database connection failed: {e}")
    
    def create_tables(self):
        create_table_query = """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(255) UNIQUE,
            original_text TEXT,
            masked_text TEXT,
            pii_mapping JSONB,
            embedding FLOAT8[]
        );
        
        CREATE INDEX IF NOT EXISTS idx_doc_id ON documents(doc_id);
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
                self.conn.commit()
        except Exception as e:
            st.error(f"Table creation failed: {e}")
    
    def store_document(self, doc_id: str, original_text: str, masked_text: str, 
                      pii_mapping: dict, embedding: np.ndarray):
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
            return True
        except Exception as e:
            st.error(f"Document storage failed: {e}")
            return False
    
    def get_similar_documents(self, query_embedding: np.ndarray, top_k: int = 5):
        select_query = """
        SELECT doc_id, masked_text, pii_mapping, embedding
        FROM documents;
        """
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(select_query)
                results = cur.fetchall()
            
            if not results:
                return []
            
            # Calculate similarities
            similarities = []
            for doc_id, masked_text, pii_mapping, embedding in results:
                doc_embedding = np.array(embedding).reshape(1, -1)
                similarity = cosine_similarity(query_embedding.reshape(1, -1), doc_embedding)[0][0]
                similarities.append({
                    'doc_id': doc_id,
                    'masked_text': masked_text,
                    'pii_mapping': json.loads(pii_mapping),
                    'similarity': similarity
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            st.error(f"Similar document retrieval failed: {e}")
            return []

class PIIProcessor:
    def __init__(self):
        self.load_models()
    
    @st.cache_resource
    def load_models(_self):
        try:
            # Load PII detection model
            pii_model = AutoModelForCausalLM.from_pretrained(
                "betterdataai/PII_DETECTION_MODEL"
            )
            pii_tokenizer = AutoTokenizer.from_pretrained(
                "betterdataai/PII_DETECTION_MODEL"
            )
            
            # Load embedding model
            embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5", trust_remote_code=True)
            
            return pii_model, pii_tokenizer, embedding_model
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            return None, None, None
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        pii_model, pii_tokenizer, _ = self.load_models()
        
        if pii_model is None:
            return {}
        
        prompt = f"""
You are an AI assistant for detecting Personal Identifiable Information (PII).

Your task is to identify and extract values from the given text based on the following 29 PII entity classes:

{', '.join(PII_CLASSES)}

Instructions:
1. Return the detected values grouped by class.
2. If no value exists for a class, skip that class.
3. Output should follow this format exactly:

<entity_class>: [value1, value2, ...]

Text:
\"\"\"{text}\"\"\"

Now extract and return the detected entities:

Output:
"""
        
        try:
            inputs = pii_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            with torch.no_grad():
                outputs = pii_model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=pii_tokenizer.eos_token_id
                )
            
            decoded = pii_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Output:" in decoded:
                decoded = decoded.split("Output:")[-1].strip()
            
            return self.parse_pii_output(decoded)
            
        except Exception as e:
            st.error(f"PII detection failed: {e}")
            return {}
    
    def parse_pii_output(self, output: str) -> Dict[str, List[str]]:
        pii_entities = {}
        lines = output.strip().split('\n')
        
        for line in lines:
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
                except:
                    continue
        
        return pii_entities
    
    def create_masked_text_and_mapping(self, text: str, pii_entities: Dict[str, List[str]]) -> Tuple[str, Dict[str, str]]:
        masked_text = text
        pii_mapping = {}
        
        # Create unique tokens for each PII value
        for entity_class, values in pii_entities.items():
            for i, value in enumerate(values):
                if value and len(value.strip()) > 0:
                    # Create unique token
                    if len(values) == 1:
                        token = entity_class
                    else:
                        token = f"{entity_class}_{i+1}"
                    
                    # Replace in text
                    masked_text = masked_text.replace(value, token)
                    pii_mapping[token] = value
        
        return masked_text, pii_mapping
    
    def generate_embedding(self, text: str) -> np.ndarray:
        _, _, embedding_model = self.load_models()
        
        if embedding_model is None:
            return np.array([])
        
        try:
            embedding = embedding_model.encode(
                [text], 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            return embedding[0]
        except Exception as e:
            st.error(f"Embedding generation failed: {e}")
            return np.array([])

class LLMProcessor:
    @staticmethod
    def query_llm(query: str, context_docs: List[Dict], combined_mapping: Dict[str, str]) -> str:
        # Prepare context from similar documents
        context = "\n\n".join([
            f"Document {i+1}: {doc['masked_text']}"
            for i, doc in enumerate(context_docs)
        ])
        
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
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"LLM query failed: {e}")
            return "Sorry, I couldn't process your request."
    
    @staticmethod
    def unmask_response(response: str, combined_mapping: Dict[str, str]) -> str:
        unmasked_response = response
        
        # Replace masked tokens with original values
        for token, original_value in combined_mapping.items():
            unmasked_response = unmasked_response.replace(token, original_value)
        
        return unmasked_response

def main():
    st.set_page_config(page_title="PII Masking & Retrieval System", layout="wide")
    
    st.title("🔒 PII Masking & Retrieval System")
    st.markdown("Upload documents to mask PII data and query them securely")
    
    # Initialize components
    db_manager = DatabaseManager()
    pii_processor = PIIProcessor()
    
    # Sidebar for mode selection
    st.sidebar.title("System Modes")
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["📤 Document Upload", "🔍 Query Documents"]
    )
    
    if mode == "📤 Document Upload":
        st.header("Document Upload & Processing")
        
        uploaded_file = st.file_uploader("Upload a text file", type=['txt'])
        
        if uploaded_file is not None:
            text_content = uploaded_file.read().decode('utf-8')
            
            st.subheader("Original Text")
            st.text_area("Content", text_content, height=200, disabled=True)
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    # Step 1: Detect PII
                    st.info("Step 1: Detecting PII entities...")
                    pii_entities = pii_processor.detect_pii(text_content)
                    
                    if pii_entities:
                        st.success(f"Detected PII entities: {list(pii_entities.keys())}")
                        
                        # Step 2: Create masked text and mapping
                        st.info("Step 2: Creating masked text and mapping...")
                        masked_text, pii_mapping = pii_processor.create_masked_text_and_mapping(
                            text_content, pii_entities
                        )
                        
                        # Step 3: Generate embedding
                        st.info("Step 3: Generating embeddings...")
                        embedding = pii_processor.generate_embedding(text_content)
                        
                        # Step 4: Store in database
                        st.info("Step 4: Storing in database...")
                        doc_id = str(uuid.uuid4())
                        success = db_manager.store_document(
                            doc_id, text_content, masked_text, pii_mapping, embedding
                        )
                        
                        if success:
                            st.success("✅ Document processed and stored successfully!")
                            
                            # Show results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Masked Text")
                                st.text_area("Masked Content", masked_text, height=200, disabled=True)
                            
                            with col2:
                                st.subheader("PII Mapping")
                                st.json(pii_mapping)
                        else:
                            st.error("Failed to store document")
                    else:
                        st.warning("No PII entities detected in the document")
    
    elif mode == "🔍 Query Documents":
        st.header("Query Processing & Retrieval")
        
        query = st.text_area("Enter your query:", height=100)
        
        if st.button("Process Query", type="primary") and query:
            with st.spinner("Processing query..."):
                # Step 1: Process query for PII
                st.info("Step 1: Processing query for PII...")
                query_pii = pii_processor.detect_pii(query)
                query_masked, query_mapping = pii_processor.create_masked_text_and_mapping(
                    query, query_pii
                )
                
                # Step 2: Generate query embedding
                st.info("Step 2: Generating query embedding...")
                query_embedding = pii_processor.generate_embedding(query)
                
                # Step 3: Find similar documents
                st.info("Step 3: Finding similar documents...")
                similar_docs = db_manager.get_similar_documents(query_embedding, top_k=5)
                
                if similar_docs:
                    st.success(f"Found {len(similar_docs)} similar documents")
                    
                    # Step 4: Combine mappings
                    combined_mapping = query_mapping.copy()
                    for doc in similar_docs:
                        combined_mapping.update(doc['pii_mapping'])
                    
                    # Step 5: Query LLM
                    st.info("Step 5: Generating response...")
                    llm_response = LLMProcessor.query_llm(query_masked, similar_docs, combined_mapping)
                    
                    # Step 6: Unmask response
                    final_response = LLMProcessor.unmask_response(llm_response, combined_mapping)
                    
                    # Display results
                    st.subheader("📊 Query Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Query Analysis")
                        st.write("**Original Query:**", query)
                        st.write("**Masked Query:**", query_masked)
                        if query_mapping:
                            st.write("**Query PII Mapping:**")
                            st.json(query_mapping)
                    
                    with col2:
                        st.subheader("Similar Documents")
                        for i, doc in enumerate(similar_docs):
                            with st.expander(f"Document {i+1} (Similarity: {doc['similarity']:.3f})"):
                                st.text(doc['masked_text'][:200] + "..." if len(doc['masked_text']) > 200 else doc['masked_text'])
                    
                    st.subheader("🤖 Final Response")
                    st.success(final_response)
                    
                else:
                    st.warning("No similar documents found")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** Make sure to set up your PostgreSQL database and Groq API key")

if __name__ == "__main__":
    main()