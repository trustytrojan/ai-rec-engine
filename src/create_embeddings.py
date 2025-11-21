import os
import argparse
import pandas as pd
import numpy as np
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_embeddings(csv_path, output_path, limit=50):
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY not set")
        return

    client = Mistral(api_key=api_key)

    print(f"Reading jobs from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Limit the number of jobs
    df = df.head(limit)
    inputs = []
    
    print(f"Processing {len(df)} jobs...")

    for _, row in df.iterrows():
        title = str(row.get('Job Title', ''))
        desc = str(row.get('Job Description', ''))
        skills = str(row.get('Skills', ''))
        text_content = f"Job Title: {title}\nDescription: {desc}\nSkills: {skills}"
        inputs.append(text_content)

    if inputs:
        print("Generating embeddings with Mistral...")
        # For larger datasets, you would want to batch these requests
        # e.g. chunks of 10 or 50
        response = client.embeddings.create(
            model="mistral-embed",
            inputs=inputs
        )
        
        # Extract embeddings
        # response.data is a list of EmbeddingObject
        embeddings = [data.embedding for data in response.data]
        
        # Convert to numpy array (float32)
        # Dimension should be 1024 for mistral-embed
        emb_array = np.array(embeddings, dtype=np.float32)
        
        print(f"Generated embeddings shape: {emb_array.shape}")
        
        # Save to binary file
        emb_array.tofile(output_path)
        print(f"Saved embeddings to {output_path}")

def _default_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    csv_file = os.path.join(data_dir, 'job_descriptions.csv')
    output_file = os.path.join(data_dir, 'job_embeddings.bin')
    return csv_file, output_file


def _parse_args():
    parser = argparse.ArgumentParser(description='Create embeddings for job CSV')
    parser.add_argument('--csv', '-c', type=str, help='Path to job CSV file (default: data/job_descriptions.csv)')
    parser.add_argument('--out', '-o', type=str, help='Output file path for embeddings binary (default: data/job_embeddings.bin)')
    parser.add_argument('--limit', '-n', type=int, default=50, help='Number of jobs to embed (default: 50)')
    return parser.parse_args()


if __name__ == "__main__":
    default_csv, default_out = _default_paths()
    args = _parse_args()
    csv_file = args.csv if args.csv else default_csv
    output_file = args.out if args.out else default_out
    create_embeddings(csv_file, output_file, limit=args.limit)
