# AI Resume Matcher

This is a Flask application that matches resumes (PDF/Text) to job descriptions using Mistral AI.

## Setup

1.  **Install Dependencies:**
    ```bash
    cd app
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configure API Key:**
    Edit `app/.env` and add your Mistral API Key:
    ```
    MISTRAL_API_KEY=your_actual_key_here
    ```

3.  **Data:**
    Ensure `job_descriptions.csv` is in the project root (`../job_descriptions.csv` relative to `app/`).

## Running

```bash
cd app
python app.py
### Run the app with custom CSV / Embeddings paths

You can override the CSV and embeddings binary used by the app at runtime:

```bash
# Example: use a trimmed CSV and small embeddings
.venv/bin/python src/app.py --csv data/job_descriptions_small.csv --embeddings data/job_embeddings_small.bin --port 5001
```

The app will print startup messages showing which CSV and embeddings were loaded.

### Extract a smaller CSV from the large zip

If you have `data/job-description-dataset.zip` (the original large archive), you can extract just the first 50 rows into `data/job_descriptions_small.csv` for faster local testing:

```bash
cd project_root
.venv/bin/python src/extract_small_csv.py
```

Then in `src/app.py`, change the CSV path to `data/job_descriptions_small.csv` if you want to run with the trimmed dataset. Or you can replace the big file with the small one (be careful to backup first).

```

## Usage

1.  Open `http://localhost:5000` in your browser.
2.  Upload a resume (PDF or Text).
3.  The app will:
    *   OCR the resume (if PDF) using Mistral OCR.
    *   Extract structured data (JSON) using Mistral Large.
    *   Generate embeddings for the resume.
    *   Match against the top 20 jobs from the CSV (limit set for demo).
    *   Return the parsed resume and top matches.

## Testing & troubleshooting

See `docs/TESTING.md` for a step-by-step guide (commands and example files) that reproduces the E2E testing we used to validate API endpoints, embeddings, and the matching logic. The doc includes fallback options if you hit Mistral rate limits and instructions for saving unique test artifacts under `data/`.

## Architecture

*   **Parsing:** Mistral OCR + Mistral Large (Chat Completion) for JSON extraction.
*   **Matching:** Mistral Embeddings + Cosine Similarity.

## API Endpoints

- POST /match
    - Upload a resume file (PDF or text) as `file`. Returns `resume_analysis` and `top_matches` where `top_matches` is a list of `{index, score}` referencing rows in `data/job_descriptions.csv`.
- GET /job/<index>
    - Returns a single job by its index (zero-based by CSV order). Use `fields` query param to limit fields, e.g. `?fields=Job Title,Company`. Default returns safe fields (no `Job Description` or `Benefits`).
- GET /jobs?ids=1,2,3
    - Returns multiple jobs by indices with `fields` param to limit returned columns.

## External resources
These are the datasets and Mistral API docs referenced in the original project prompt and used while building/testing the system.

- Job listings dataset (Kaggle): https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset/data
- Resumes dataset (Hugging Face): https://huggingface.co/datasets/datasetmaster/resumes
- Mistral Document Annotations API (Document AI): https://docs.mistral.ai/capabilities/document_ai/annotations
- Mistral Text Embeddings API: https://docs.mistral.ai/capabilities/embeddings/text_embeddings

Please consult these links for dataset schema details and Mistral API usage when generating embeddings and parsing documents.
