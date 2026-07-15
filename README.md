## LLM Hallucination Detection MVP

This project is a small prototype for detecting potential **LLM hallucinations** in high-stakes domains (finance, law, healthcare).  
It accepts a question and an LLM answer, decomposes the answer into atomic factual claims, verifies them against domain knowledge bases, and returns a calibrated hallucination risk score plus a detailed JSON report.

### Requirements

- Python 3.10+
- An OpenAI API key (optional but recommended) exposed as `OPENAI_API_KEY`.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the API server

Set your OpenAI key (if you want real LLM calls):

```bash
export OPENAI_API_KEY="sk-..."
```

Then start the FastAPI server with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The minimal web UI is available at:

- `http://localhost:8000/` – interactive page for running analyses

Health check:

- `http://localhost:8000/health`

### Using the `/analyze` endpoint

Endpoint:

- `POST /analyze`

Example JSON body:

```json
{
  "domain": "healthcare",
  "question": "What are the risks of untreated hypertension?",
  "answer": null,
  "use_mock_llm": false
}
```

- If `answer` is `null` or omitted, the backend will call OpenAI to generate an answer.
- If you provide `answer`, the backend will skip generation and only analyze it.
- If `use_mock_llm` is `true`, the backend will not call OpenAI and will use a simple mock LLM instead.

The response includes:

- The domain, question, and answer
- A list of claims with:
  - Evidence snippets from the toy knowledge base
  - A label (`supported`, `contradicted`, `unknown`)
  - An NLI-style relation (`entailment`, `contradiction`, `neutral`)
  - An evidence score \[0, 1\]
  - Any confidence–evidence mismatch flags
- An overall hallucination score and explanation string
- `score_breakdown` with normalized ratios and calibrated score
- `analysis_details` showing:
  - generated vs pasted answer source
  - extracted claims
  - matched evidence snippets
  - scoring rationale

### Accuracy updates included

- Length-bias fixed with claim-count-normalized scoring (weighted ratios, not raw additive sum).
- Retrieval improved with hybrid keyword + semantic similarity scoring.
- NLI-style contradiction inference added for stronger claim verification.
- Score calibration added using labeled toy samples in `data/calibration_samples.json`.
- UI transparency panel added to show what is analyzed in the background.

### Running tests

```bash
pytest
```

### Notes and limitations

- The knowledge bases in `data/finance.json`, `data/law.json`, and `data/healthcare.json` are **toy examples**, not real production datasets.
- Claim decomposition and verification are heuristic and meant for demonstration, not for clinical, legal, or financial use.
- Do not rely on this system for real-world decisions in high-stakes contexts.

