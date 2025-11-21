import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from mistralai import Mistral
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Mistral Client
api_key = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=api_key) if api_key else None

def get_text_embedding(text: str) -> List[float]:
    """
    Generates an embedding for a single string of text using Mistral's embedding API.
    """
    if not client:
        raise ValueError("Mistral API Key not set")
    
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[text]
    )
    return response.data[0].embedding

def load_job_data(csv_path: str, bin_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Loads job descriptions from CSV and pre-computed embeddings from a binary file.
    Returns a tuple of (DataFrame, Embeddings Matrix).
    """
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
        # Normalize and reset index so DataFrame row indices correspond to our indices
        df = df.reset_index(drop=True)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return pd.DataFrame(), np.array([])

    # Read Embeddings
    try:
        # Load raw float32 data
        embeddings = np.fromfile(bin_path, dtype=np.float32)
        # Reshape to (-1, 1024) as Mistral embeddings are 1024d
        embeddings = embeddings.reshape(-1, 1024)
    except Exception as e:
        print(f"Error reading embeddings file: {e}")
        return df, np.array([])

    # Ensure we only keep as many CSV rows as we have embeddings
    # (In case the CSV is larger than the generated embeddings limit)
    if len(df) > len(embeddings):
        df = df.iloc[:len(embeddings)]
    
    return df, embeddings

def extract_resume_json(text_content: str) -> Dict[str, Any]:
    """
    Uses Mistral Chat Completion to parse raw resume text into structured JSON.
    """
    if not client:
        raise ValueError("Mistral API Key not set")

    prompt = """
    You are an expert resume parser. Extract the following information from the resume text below and return it as a valid JSON object.
    
    Fields to extract:
    - personal_information (name, email, phone, location)
    - experience (list of objects with title, company, dates, description)
    - education (list of objects with degree, institution, year)
    - skills (list of strings)
    - projects (list of objects with name, description)

    Resume Text:
    {text}
    
    Return ONLY the JSON object.
    """
    
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "user", "content": prompt.format(text=text_content)}
        ],
        response_format={"type": "json_object"}
    )
    
    try:
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {"raw_text": text_content}

def process_resume_file(file_path: str, file_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Processes a resume file (PDF/Image/Text).
    If PDF/Image, uses Mistral OCR.
    If Text, reads directly.
    Returns structured JSON.
    """
    if not client:
        raise ValueError("Mistral API Key not set")

    extracted_text = ""

    if file_path.lower().endswith('.pdf'):
        # Use Mistral OCR
        # For local files, we need to upload or encode. 
        # The SDK supports passing a file directly if we use the upload method, 
        # or base64 for the process method.
        
        import base64
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_base64",
                "document_base64": encoded_string
            }
        )
        # Combine all pages markdown
        extracted_text = "\n".join([page.markdown for page in ocr_response.pages])
        
    else:
        # Assume text file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            extracted_text = f.read()

    # Now convert to JSON
    resume_json = extract_resume_json(extracted_text)
    return resume_json

def match_resume_to_jobs(resume_json: Dict[str, Any], job_df: pd.DataFrame, job_embeddings: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Matches a structured resume against job embeddings.
    """
    # Create a text representation of the resume for embedding
    # We focus on Skills, Experience, and Projects for matching
    skills = ", ".join(resume_json.get('skills', []))
    
    exp_text = ""
    for exp in resume_json.get('experience', []):
        exp_text += f"{exp.get('title', '')} at {exp.get('company', '')}. {exp.get('description', '')} "
        
    text_for_embedding = f"Skills: {skills}\nExperience: {exp_text}"
    
    resume_vector = get_text_embedding(text_for_embedding)
    
    # Calculate similarities
    if len(job_embeddings) == 0:
        return []

    # cosine_similarity computes dot product of normalized vectors
    # shape: (1, N)
    sims = cosine_similarity([resume_vector], job_embeddings)[0]
    
    # Get top K indices
    # argsort returns indices that would sort the array
    # we take the last top_k (highest scores) and reverse them
    top_indices = sims.argsort()[-top_k:][::-1]
    
    # Return indices and scores only. This keeps the match response compact
    results = []
    for idx in top_indices:
        results.append({
            "index": int(idx),
            "score": float(sims[idx])
        })
    
    return results
