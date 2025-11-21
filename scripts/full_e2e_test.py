#!/usr/bin/env python3
"""
Orchestrate an end-to-end test for the AI Resume Matcher.

This script will:
- create a `data/` directory if missing
- download the HuggingFace resumes dataset (if possible)
- download the Kaggle jobs dataset using `kaggle` if available
- extract a subset CSV using `src/extract_small_csv.py`
- generate embeddings using `src/create_embeddings.py` (Mistral API) or fallback random embeddings
- start the Flask app pointing to the created CSV and embeddings on a test port
- post a random resume to `/match` and print the top match index
- GET `/job/<index>` for the top match and print limited fields

This script is helpful for reproducing the TESTING.md steps and automating the workflow.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
import signal
from pathlib import Path
import random
import json
import shutil

from huggingface_hub import hf_hub_download

# Ensure project root on sys.path so imports from `scripts` package succeed when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_small_csv import extract_small_csv
from scripts.create_embeddings import create_embeddings


def run(cmd, check=True, capture_output=False, env=None):
    print("RUN:", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, env=env)


def ensure_data_dir(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data dir: {data_dir}")


def download_hf_resume(data_dir: Path):
    # Try to download the master_resumes.jsonl from HuggingFace dataset
    repo_id = "datasetmaster/resumes"
    filename = "master_resumes.jsonl"
    out = data_dir / filename
    if out.exists():
        print(f"Found existing {out}")
        return out
    try:
        print(f"Attempting to download {repo_id}/{filename} to {out}")
        hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", local_dir=str(data_dir))
        if not out.exists():
            raise FileNotFoundError(f"Downloaded to {data_dir}, but {filename} missing")
        print("Downloaded HuggingFace resumes dataset")
        return out
    except Exception as e:
        print("Warning: failed to download from HuggingFace automatically:", e)
        print("Please ensure you have access or download manually and place in data/ as master_resumes.jsonl")
        return out


def download_kaggle_jobs(data_dir: Path):
    # Try to use kaggle CLI to download the dataset (it requires credentials/cli installed)
    zip_name = data_dir / "job-description-dataset.zip"
    csv_name = data_dir / "job_descriptions.csv"
    if csv_name.exists() or zip_name.exists():
        print(f"Found existing jobs file: {csv_name if csv_name.exists() else zip_name}")
        return csv_name if csv_name.exists() else zip_name
    # Check kaggle CLI presence
    try:
        run(["kaggle", "--version"], check=True, capture_output=True)
    except Exception:
        print("Kaggle CLI is not available or not configured. Skipping Kaggle download.")
        print("You can download the dataset manually from https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset/data and place the ZIP in data/ with the name job-description-dataset.zip")
        return None

    try:
        # Download dataset using kaggle CLI and unzip
        run(["kaggle", "datasets", "download", "-d", "ravindrasinghrana/job-description-dataset", "-p", str(data_dir), "--unzip"])
        # Verify csv exists
        if csv_name.exists():
            print("Downloaded and unzipped Kaggle job dataset")
            return csv_name
        else:
            print("Dataset downloaded but CSV not found; check data folder")
            return None
    except Exception as e:
        print("Error downloading Kaggle dataset:", e)
        return None


def pick_random_resume(master_jsonl: Path, out_resume: Path):
    lines = master_jsonl.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError("No lines in master_resumes.jsonl")
    idx = random.randrange(len(lines))
    out_resume.write_text(lines[idx], encoding="utf-8")
    print(f"Wrote random resume to {out_resume} (index {idx})")
    return idx


def extract_jobs(data_dir: Path, zip_csv: Path | None, out_csv: Path, limit: int = 150, start: int = 0):
    # if zip_csv is a zip -> call script with zip path; else if csv already exists -> copy to out_csv
    if zip_csv is None:
        raise RuntimeError("No job CSV or zip file available to extract from")
    if zip_csv.suffix == ".zip":
        # call extract_small_csv directly from the module
        extract_small_csv(str(zip_csv), "job_descriptions.csv", str(out_csv), limit=limit, start=start)
    else:
        # It's a CSV already; just copy rows
        if not zip_csv.exists():
            raise FileNotFoundError(f"CSV not found: {zip_csv}")
        # simply copy and head - take header + slice
        import csv as _csv
        with open(zip_csv, 'r', encoding='utf-8', errors='ignore') as f_in, open(out_csv, 'w', newline='', encoding='utf-8') as f_out:
            reader = _csv.reader(f_in)
            writer = _csv.writer(f_out)
            header = next(reader)
            writer.writerow(header)
            # skip start
            for _ in range(start):
                try:
                    next(reader)
                except StopIteration:
                    break
            count = 0
            for row in reader:
                writer.writerow(row)
                count += 1
                if count >= limit:
                    break
        print(f"Wrote {count} rows (plus header) to {out_csv}")
    return out_csv


def create_embeddings_or_fallback(csv_path: Path, out_bin: Path, limit: int = 150):
    # Try to run the create_embeddings.py script
    try:
        # Call the create_embeddings function directly
        create_embeddings(str(csv_path), str(out_bin), limit=limit)
        if out_bin.exists():
            print("create_embeddings() wrote embeddings successfully")
            return out_bin
        else:
            print("create_embeddings() did not produce the expected output; falling back to random embeddings")
    except Exception as e:
        print("create_embeddings() raised an exception:", e)
        print("Falling back to random embeddings due to Mistral unavailability or errors")
    except Exception as e:
        print("Error running create_embeddings.py:", e)

    # fallback: create random embeddings (shape: n_jobs, 1024)
    import numpy as np
    n_jobs = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
    arr = np.random.rand(n_jobs, 1024).astype('float32')
    arr.tofile(out_bin)
    print(f"Wrote fallback embeddings to {out_bin}")
    return out_bin


def start_flask_app(csv_path: Path, bin_path: Path, port: int = 5001):
    log_path = Path("/tmp") / f"flask_test_{int(time.time())}.log"
    env = os.environ.copy()
    # Launch app in background
    cmd = [sys.executable, "src/app.py", "--csv", str(csv_path), "--embeddings", str(bin_path), "--host", "127.0.0.1", "--port", str(port)]
    print("Starting Flask app:", cmd)
    proc = subprocess.Popen(cmd, stdout=open(log_path, 'w'), stderr=subprocess.STDOUT, env=env)
    # wait for the app to start by tailing logs
    print("Flask log path:", log_path)
    started = False
    for _ in range(20):
        if log_path.exists():
            text = log_path.read_text(errors='ignore')
            if "Loaded" in text and "jobs" in text:
                started = True
                break
            if "Running on" in text:
                started = True
                break
        time.sleep(0.5)
    if not started:
        print("Warning: Flask app might not be ready. Check logs:", log_path)
    return proc, log_path


def post_resume_and_get_top_match(resume_file: Path, port: int = 5001):
    import requests
    url = f"http://127.0.0.1:{port}/match"
    with open(resume_file, 'rb') as fh:
        files = {'file': (resume_file.name, fh, 'application/json')}
        r = requests.post(url, files=files)
    if r.status_code != 200:
        print("POST /match returned status", r.status_code, r.text)
        # Return the response object so the caller can inspect the error
        return r
    data = r.json()
    print(json.dumps(data, indent=2)[:2000])
    top_idx = None
    try:
        top_idx = data['top_matches'][0]['index']
    except Exception:
        print('Could not find top match')
    return top_idx


def get_job_details(job_idx: int, port: int = 5001):
    import requests
    url = f"http://127.0.0.1:{port}/job/{job_idx}?fields=Job Title,Company,Job Description"
    r = requests.get(url)
    if r.status_code != 200:
        print("GET /job/ returned status", r.status_code, r.text)
        return None
    return r.json()


def fallback_match_from_local(resume_file: Path, job_csv: Path, embeddings_bin: Path, top_k: int = 5):
    """Fallback matching logic when Mistral APIs are unavailable.

    We'll compute a random resume vector and compute cosine similarity vs the job embeddings bin.
    """
    import numpy as np
    # Load job embeddings
    data = np.fromfile(str(embeddings_bin), dtype=np.float32)
    if data.size == 0:
        print('Embeddings file is empty. Cannot fallback match.')
        return []
    emb_matrix = data.reshape(-1, 1024)
    n = emb_matrix.shape[0]
    # Use a random vector for resume embedding
    resume_vec = np.random.rand(1024).astype(np.float32)
    # cosine similarity
    import sklearn.metrics.pairwise as pw
    sims = pw.cosine_similarity([resume_vec], emb_matrix)[0]
    indices = sims.argsort()[-top_k:][::-1]
    return [(int(i), float(sims[i])) for i in indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets and run the full end-to-end test for AI Resume Matcher")
    parser.add_argument('--data-dir', default='data', help='Where to put downloaded datasets and test artifacts')
    parser.add_argument('--jobs-limit', type=int, default=150, help='Number of jobs to extract into test CSV')
    parser.add_argument('--jobs-start', type=int, default=0, help='Start index for the CSV extraction')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the Flask server on')
    parser.add_argument('--no-kaggle', action='store_true', help='do not attempt to use kaggle CLI')
    parser.add_argument('--resume-index', type=int, default=None, help='If provided, use this line number of master_resumes.jsonl instead of random')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    ensure_data_dir(data_dir)

    # 1) Hugging Face dataset
    master_resumes = download_hf_resume(data_dir)

    # 2) Kaggle dataset
    job_source = None
    if not args.no_kaggle:
        job_source = download_kaggle_jobs(data_dir)
    if job_source is None:
        print("Kaggle dataset not downloaded automatically; trying to find a local CSV in data/")
        # try to detect CSV in data
        csv_candidates = list(data_dir.glob("*.csv"))
        if csv_candidates:
            job_source = csv_candidates[0]
            print("Found local CSV candidate:", job_source)
        else:
            print("No CSV found; aborting")
            return

    # 3) Pick a resume
    ts = int(time.time())
    resume_file = data_dir / f"test_resume_{ts}.json"
    if args.resume_index is not None:
        # pick a specific line
        lines = master_resumes.read_text(encoding='utf-8').splitlines()
        idx = args.resume_index
        resume_file.write_text(lines[idx], encoding='utf-8')
    else:
        idx = pick_random_resume(master_resumes, resume_file)
    print("Using resume file:", resume_file)

    # 4) extract jobs to data/test_jobs_${ts}.csv
    out_csv = data_dir / f"test_jobs_{ts}.csv"
    extract_jobs(data_dir, job_source, out_csv, limit=args.jobs_limit, start=args.jobs_start)

    # 5) create embeddings
    out_bin = data_dir / f"test_embeddings_{ts}.bin"
    embeddings_file = create_embeddings_or_fallback(out_csv, out_bin, limit=args.jobs_limit)

    # 6) start Flask app
    proc, log_path = start_flask_app(out_csv, embeddings_file, port=args.port)

    try:
        # 7) post the resume and get top index
        post_res = post_resume_and_get_top_match(resume_file, port=args.port)
        top_idx = None
        if hasattr(post_res, 'status_code'):
            # It was an HTTP response due to an error
            txt = post_res.text
            if 'Service tier capacity exceeded' in txt or post_res.status_code >= 500:
                # fallback to local matching
                print('Mistral unavailable or error in /match; falling back to local matching')
                top_matches = fallback_match_from_local(resume_file, out_csv, embeddings_file, top_k=5)
                print('Top matches (fallback):', top_matches)
                if top_matches:
                    top_idx = top_matches[0][0]
            else:
                print('Unexpected error from POST /match:')
                print(txt)
        else:
            top_idx = post_res
            print('Top match index:', top_idx)
            details = get_job_details(top_idx, port=args.port)
            print('Job details:')
            print(json.dumps(details, indent=2))
    finally:
        # Clean up: stop Flask app
        print('Stopping Flask PID', proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        print('Flask stopped')


if __name__ == '__main__':
    main()
