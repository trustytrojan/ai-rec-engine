# Testing Workflow — AI Resume Matcher

This document reproduces the end-to-end testing steps and commands used in the repository to verify the resume matcher.

It uses the following resources under the project `data/` directory (unique filenames are used in the example):
- `data/master_resumes.jsonl` — many resumes (JSONL)
- `data/job-description-dataset.zip` — large CSV archive of job descriptions

Test files used in the example:
 - `data/test_resume.json` — randomly selected resume saved to file
 - `data/test_jobs.csv` — extracted subset of job CSV (150 rows)
 - `data/test_embeddings.bin` — embeddings for the above CSV (or a fallback file if Mistral API was limited)

## Prerequisites
- Python 3.13+ (or your Python venv variant)
- A virtual environment with packages installed (see `requirements.txt`)
- Optional: an active Mistral API Key in `MISTRAL_API_KEY` for embeddings and OCR

Quick setup (if not already done):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Pick a random resume and save it
This saves the first (or a random) line from `master_resumes.jsonl` to `data/` as a JSON file.

```bash
python - <<'PY'
import random, time
from pathlib import Path
p = Path('data/master_resumes.jsonl')
lines = p.read_text(encoding='utf-8').splitlines()
idx = random.randrange(len(lines))
content = lines[idx]
fn = Path('data') / "test_resume.json"
fn.write_text(content, encoding='utf-8')
print('Wrote', fn)
PY
```

- You can also just take the first line with the `head` util or a one-liner.

## 2) Extract a chunk of jobs from the ZIP (use `extract_small_csv.py`)
The `extract_small_csv.py` script extracts a subset (header + data rows) from the large job CSV inside the ZIP.

Example: extract 150 rows and write to `data/test_jobs.csv`:

```bash
# Extract jobs to a fixed filename
python src/extract_small_csv.py --zip data/job-description-dataset.zip --inner job_descriptions.csv --out data/test_jobs.csv --limit 150 --start 0
# Verify the file exists and has 151 lines (header + 150 rows)
wc -l data/test_jobs.csv
```

Notes:
- `--start` is 0-based and counts data rows (the header isn't counted). `--limit` is the number of data rows to extract.

## 3) Create embeddings with `create_embeddings.py`
If Mistral is available and you have the key set, generate embeddings for your extracted CSV:

```bash
# export MISTRAL_API_KEY=your_key
.venv/bin/python src/create_embeddings.py --csv data/test_jobs.csv --out data/test_embeddings.bin --limit 150
```

If Mistral is rate-limited or not available, create fallback random embeddings locally (quick & reproducible):

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
csv_path = Path('data/test_jobs.csv')
# Count rows minus header
n_jobs = sum(1 for _ in open(csv_path, 'r', encoding='utf-8')) - 1
emb_path = Path('data/test_embeddings_fallback.bin')
arr = np.random.rand(n_jobs, 1024).astype('float32')
arr.tofile(emb_path)
print('Wrote fallback embeddings to', emb_path)
PY
```

Caveat: random embeddings won't produce semantically meaningful matches; they're for testing the end-to-end system only.

## 4) Start the Flask app (point it to the CSV and the embeddings)
Start the Flask app on port 5001 and confirm it loads the CSV and embedding files:

```bash
# Stop any running dev server
pkill -f "src/app.py" || true
# Start Flask in background (see /tmp/flask_test_5001.log for logs)
.venv/bin/python src/app.py --csv data/test_jobs.csv --embeddings data/test_embeddings.bin --host 127.0.0.1 --port 5001 &> /tmp/flask_test_5001.log & echo $!
# Wait briefly then tail the logs to confirm startup and loading
sleep 1
sed -n '1,200p' /tmp/flask_test_5001.log
```

You should see:
- A message indicating the CSV and bin were loaded, e.g. `Loaded 150 jobs and embeddings matrix of shape (150, 1024).`
- `Running on http://127.0.0.1:5001`

## 5) POST the resume to `/match` (the server will parse the resume and match it)
Use a simple `curl` request to upload the JSON or text resume file (it’s handled via the file form field):

```bash
curl -i -s -X POST -F "file=@data/test_resume.json;type=application/json" http://127.0.0.1:5001/match | jq .
```

What to expect in the response
- `resume_analysis`: Contains parsed fields (education, experience, skills, etc.) returned by the parser.
- `top_matches`: A list of objects `{index, score}` showing indices into the CSV you supplied earlier (0-based row indices).

## 6) Fetch the top match job details with `/jobs/<index>`
Take the top match index from the `/match` response and fetch the job details:

```bash
# Example where top match index is 1
curl -s "http://127.0.0.1:5001/jobs/1" | jq .
```

This returns the job details as a JSON object.

## 7) Troubleshooting & Notes
- If `create_embeddings.py` fails with a 429 (Service tier capacity exceeded), wait for Mistral cooldown or use the fallback random embeddings step above.
- Ensure `MISTRAL_API_KEY` is set in the environment for non-fallback runs.
- Use the venv's Python (`.venv/bin/python`) for consistent environment and package versions.
- The `--start` and `--limit` options on `extract_small_csv.py` make it easy to create reproducible, unique test datasets.

## Reproducible quick commands summary
```bash
# Setup
python -m venv .venv; source .venv/bin/activate; pip install -r requirements.txt

# 1 - select resume
python - <<'PY'
from pathlib import Path
fn='data/test_resume.json'
Path(fn).write_text(Path('data/master_resumes.jsonl').read_text().splitlines()[0])
print('Wrote', fn)
PY

# 2 - extract jobs
python src/extract_small_csv.py --zip data/job-description-dataset.zip --inner job_descriptions.csv --out data/test_jobs.csv --limit 150 --start 0

# 3 - create embeddings (or fallback)
.venv/bin/python src/create_embeddings.py --csv data/test_jobs.csv --out data/test_embeddings.bin --limit 150
# OR fallback
python - <<'PY'
# fallback random embeddings
import numpy as np
from pathlib import Path
csv_path=Path('data/test_jobs.csv')
rows=sum(1 for _ in open(csv_path))-1
arr=np.random.rand(rows,1024).astype('float32')
arr.tofile(Path('data/test_embeddings_fallback.bin'))
PY

# 4 - start app
.venv/bin/python src/app.py --csv data/test_jobs.csv --embeddings data/test_embeddings.bin --host 127.0.0.1 --port 5001 &> /tmp/flask_test_5001.log &

# 5 - post resume and get top match
curl -s -X POST -F "file=@data/test_resume.json;type=application/json" http://127.0.0.1:5001/match | jq .

# 6 - fetch job details
curl -s "http://127.0.0.1:5001/jobs/1" | jq .
```

## Wrap-up
This testing steps document aims to capture the live exploratory test we ran for the AI Resume Matcher project and makes the process reproducible. If you'd like, I can extend this to a small script (bash or python) that runs the whole flow and captures outputs into an artifacts folder for easier regression testing.

