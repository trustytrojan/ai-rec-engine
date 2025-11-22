import os
import argparse
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils import load_job_data, process_resume_file, match_resume_to_jobs

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global storage for job data
JOB_DF = None
JOB_EMBEDDINGS_MATRIX = None

def load_data():
    global JOB_DF, JOB_EMBEDDINGS_MATRIX
    # Project root is the parent of the src directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    csv_path = os.path.join(data_dir, 'job_descriptions_small.csv')
    bin_path = os.path.join(data_dir, 'job_embeddings.bin')
    
    if os.path.exists(csv_path) and os.path.exists(bin_path):
        print("Loading job data...")
        JOB_DF, JOB_EMBEDDINGS_MATRIX = load_job_data(csv_path, bin_path)
        print(f"Loaded {len(JOB_DF)} jobs and embeddings matrix of shape {JOB_EMBEDDINGS_MATRIX.shape}.")
    else:
        print(f"Warning: Data files not found. \nCSV: {csv_path} ({os.path.exists(csv_path)})\nBin: {bin_path} ({os.path.exists(bin_path)})")

# We will load data as part of startup only once we parse arguments.
# If you call `python src/app.py` directly, the defaults below will be used.

@app.route('/', methods=['GET'])
def index():
    return render_template_string('''
        <!doctype html>
        <title>Resume Matcher</title>
        <h1>Upload Resume to Match Jobs</h1>
        <form method=post action="/match" enctype=multipart/form-data>
          <input type=file name=file>
          <input type=submit value=Upload>
        </form>
    ''')

@app.route('/match', methods=['POST'])
def match_resume():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # 1. Process Resume (OCR -> JSON)
            resume_json = process_resume_file(filepath)
            
            # 2. Match against loaded jobs
            if JOB_DF is None or JOB_EMBEDDINGS_MATRIX is None:
                return jsonify({
                    "message": "No job embeddings loaded. Check API key and data files.",
                    "resume_parsed": resume_json
                })
                
            matches = match_resume_to_jobs(resume_json, JOB_DF, JOB_EMBEDDINGS_MATRIX)
            
            return jsonify({
                "resume_analysis": resume_json,
                "top_matches": matches
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)

def _normalize_column_name(col: str) -> str:
    # Helper to create a normalization key for column matching
    return ''.join(ch for ch in col.lower() if ch.isalnum())


def _map_fields_to_columns(df: 'pd.DataFrame', requested_fields: list[str]) -> list[str]:
    # Build map from normalized name to col
    col_map = { _normalize_column_name(col): col for col in df.columns }
    mapped = []
    for f in requested_fields:
        key = _normalize_column_name(f)
        if key in col_map:
            mapped.append(col_map[key])
    return mapped


def _to_python_value(v):
    import numpy as _np
    import pandas as _pd
    # Handle NaN / NA
    try:
        if _pd.isna(v):
            return None
    except Exception:
        pass
    # numpy scalars
    if isinstance(v, (_np.integer, _np.floating, _np.bool_)):
        return v.item()
    if isinstance(v, _np.ndarray):
        return v.tolist()
    # Fallback
    return v


@app.route('/jobs/<int:idx>', methods=['GET'])
def get_job_by_index(idx: int):
    """Return the full job JSON for a single job index. No `fields` param supported.

    Example: GET /jobs/123
    """
    if JOB_DF is None:
        return jsonify({"error": "Job data not loaded"}), 500

    if idx < 0 or idx >= len(JOB_DF):
        return jsonify({"error": "Job index out of range"}), 404

    job_row = JOB_DF.iloc[idx]
    result = {'index': int(idx)}
    for col in JOB_DF.columns:
        result[col] = _to_python_value(job_row.get(col))

    return jsonify(result)


def _parse_args():
    parser = argparse.ArgumentParser(description='Run resume matcher Flask app')
    parser.add_argument('--csv', '-c', type=str, help='Path to job descriptions CSV (defaults to data/job_descriptions_small.csv)')
    parser.add_argument('--embeddings', '-e', type=str, help='Path to job embeddings binary (defaults to data/job_embeddings.bin)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to bind Flask server to')
    parser.add_argument('--port', '-p', type=int, default=5000, help='Port for Flask server')
    return parser.parse_args()


if __name__ == '__main__':
    # Default paths derived from project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    default_csv = os.path.join(data_dir, 'job_descriptions_small.csv')
    default_bin = os.path.join(data_dir, 'job_embeddings.bin')

    args = _parse_args()
    csv_path = args.csv if args.csv else default_csv
    bin_path = args.embeddings if args.embeddings else default_bin
    # Set up global variables (used by endpoints)
    load_data_with_paths = getattr(load_data, '__call__', None)
    # Call load_data with our specific paths: monkeypatch behavior by redefining
    def load_data_with_override():
        global JOB_DF, JOB_EMBEDDINGS_MATRIX
        if os.path.exists(csv_path) and os.path.exists(bin_path):
            print(f"Loading job data from: CSV='{csv_path}', BIN='{bin_path}'")
            JOB_DF, JOB_EMBEDDINGS_MATRIX = load_job_data(csv_path, bin_path)
            print(f"Loaded {len(JOB_DF)} jobs and embeddings matrix of shape {JOB_EMBEDDINGS_MATRIX.shape}.")
        else:
            print(f"Warning: Data files not found. \nCSV: {csv_path} ({os.path.exists(csv_path)})\nBin: {bin_path} ({os.path.exists(bin_path)})")

    # Load with override paths
    load_data_with_override()

    # start Flask
    app.run(host=args.host, port=args.port)
