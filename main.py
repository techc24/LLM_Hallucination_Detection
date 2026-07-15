import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from .claims import rule_based_claim_split
from .llm_client import LLMClient
from .schemas import AnalyzeRequest, AnalyzeResponse, ClaimResult, Domain
from .scoring import aggregate_hallucination
from .verification import detect_confidence_flags, score_claim


app = FastAPI(title="LLM Hallucination Detection MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root_ui() -> str:
    """
    Minimal UI: simple HTML page that calls /analyze and shows JSON.
    """
    return """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>LLM Hallucination Detector</title>
    <style>
      body { font-family: system-ui, -apple-system, sans-serif; margin: 2rem; background: #0f172a; color: #e5e7eb; }
      .card { max-width: 960px; margin: 0 auto; background: #020617; padding: 1.5rem 2rem; border-radius: 0.75rem; box-shadow: 0 20px 25px -20px rgba(15,23,42,0.8); border: 1px solid #1f2937; }
      h1 { font-size: 1.6rem; margin-bottom: 0.5rem; }
      p.sub { color: #9ca3af; margin-bottom: 1rem; }
      label { display: block; margin-top: 0.75rem; font-weight: 500; }
      select, textarea, input { width: 100%; margin-top: 0.25rem; padding: 0.5rem 0.6rem; border-radius: 0.5rem; border: 1px solid #374151; background: #020617; color: #e5e7eb; font-family: inherit; font-size: 0.95rem; }
      textarea { min-height: 80px; resize: vertical; }
      button { margin-top: 1rem; padding: 0.6rem 1.2rem; border-radius: 999px; border: none; background: linear-gradient(to right, #22c55e, #0ea5e9); color: #020617; font-weight: 600; cursor: pointer; }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      pre { margin-top: 1rem; padding: 0.75rem 1rem; background: #020617; border-radius: 0.5rem; border: 1px solid #111827; max-height: 360px; overflow: auto; font-size: 0.85rem; }
      .panel { margin-top: 0.75rem; padding: 0.75rem 1rem; background: #020617; border-radius: 0.5rem; border: 1px solid #111827; }
      .muted { color: #9ca3af; font-size: 0.9rem; margin-top: 0.35rem; }
      .score { margin-top: 0.75rem; font-weight: 600; }
      .badge { display: inline-block; padding: 0.15rem 0.6rem; font-size: 0.7rem; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.04em; }
      .badge-low { background: rgba(34,197,94,0.12); color: #4ade80; }
      .badge-med { background: rgba(234,179,8,0.12); color: #facc15; }
      .badge-high { background: rgba(248,113,113,0.12); color: #fca5a5; }
      .row { display: flex; flex-direction: column; gap: 0.75rem; margin-top: 0.5rem; }
      .row > div { flex: 1; }
      small { color: #6b7280; display: block; margin-top: 0.25rem; }
    </style>
  </head>
  <body>
    <div class="card">
      <h1>LLM Hallucination Detection</h1>
      <p class="sub">Analyze answers in finance, law, or healthcare for potential hallucinations using toy knowledge bases.</p>

      <label>Domain</label>
      <select id="domain">
        <option value="finance">Finance</option>
        <option value="law">Law</option>
        <option value="healthcare">Healthcare</option>
      </select>

      <label>Question</label>
      <textarea id="question" placeholder="Enter a user question..."></textarea>

      <div class="row">
        <div>
          <label>Answer (optional)</label>
          <textarea id="answer" placeholder="Paste a model answer, or leave blank to let the backend call OpenAI."></textarea>
        </div>
        <div>
          <label>Use mock LLM?</label>
          <select id="use_mock_llm">
            <option value="false">No (use OpenAI)</option>
            <option value="true">Yes (no API calls)</option>
          </select>
          <small>Mock mode skips OpenAI; useful if no key is set.</small>
        </div>
      </div>

      <button id="runBtn" onclick="runAnalyze()">Run analysis</button>
      <div class="score" id="score"></div>
      <div class="panel">
        <strong>Answer being analyzed</strong>
        <div class="muted" id="answerSource">Source: -</div>
        <pre id="analyzedAnswer"></pre>
      </div>
      <div class="panel">
        <strong>Background analysis details</strong>
        <pre id="details"></pre>
      </div>
      <pre id="output"></pre>
    </div>

    <script>
      async function runAnalyze() {
        const btn = document.getElementById("runBtn");
        const domain = document.getElementById("domain").value;
        const question = document.getElementById("question").value;
        const answer = document.getElementById("answer").value;
        const useMock = document.getElementById("use_mock_llm").value === "true";
        const out = document.getElementById("output");
        const details = document.getElementById("details");
        const analyzedAnswer = document.getElementById("analyzedAnswer");
        const answerSource = document.getElementById("answerSource");
        const scoreDiv = document.getElementById("score");

        if (!question.trim()) {
          alert("Please enter a question.");
          return;
        }

        btn.disabled = true;
        out.textContent = "Running analysis...";
        details.textContent = "Collecting generated/pasted answer, extracted claims, evidence snippets, and score breakdown...";
        analyzedAnswer.textContent = "Preparing answer for analysis...";
        answerSource.textContent = "Source: calculating...";
        scoreDiv.textContent = "";

        try {
          const res = await fetch("/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ domain, question, answer: answer || null, use_mock_llm: useMock })
          });
          const raw = await res.text();
          let data = null;
          try {
            data = JSON.parse(raw);
          } catch (parseErr) {
            throw new Error(raw || "Server returned a non-JSON response.");
          }
          if (!res.ok) {
            const detail = (data && data.detail) ? data.detail : JSON.stringify(data);
            throw new Error(detail);
          }
          out.textContent = JSON.stringify(data, null, 2);
          details.textContent = JSON.stringify(data.analysis_details || {}, null, 2);
          analyzedAnswer.textContent = (data && data.answer) ? data.answer : "(No answer text returned)";
          const src = data && data.analysis_details && data.analysis_details.answer_source
            ? data.analysis_details.answer_source
            : "unknown";
          answerSource.textContent = "Source: " + src;

          if (data && typeof data.hallucination_score === "number") {
            const s = data.hallucination_score;
            let badgeClass = "badge-low";
            let label = "Low risk";
            if (s >= 0.66) { badgeClass = "badge-high"; label = "High risk"; }
            else if (s >= 0.33) { badgeClass = "badge-med"; label = "Medium risk"; }
            scoreDiv.innerHTML = 'Hallucination score: ' + s.toFixed(2) + ' ' +
              '<span class="badge ' + badgeClass + '">' + label + '</span>';
          }
        } catch (err) {
          out.textContent = "Error: " + err;
          analyzedAnswer.textContent = "Answer unavailable due to request error.";
          answerSource.textContent = "Source: unavailable";
        } finally {
          btn.disabled = false;
        }
      }
    </script>
  </body>
</html>
    """

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    use_mock = req.use_mock_llm or bool(os.getenv("USE_MOCK_LLM", "").lower() == "true")
    llm = LLMClient(use_mock=use_mock)
    try:
        if req.answer:
            answer = req.answer
        else:
            answer = llm.generate_answer(req.domain.value, req.question)

        claims: List[str] = llm.decompose_claims(answer)
        if not claims:
            claims = rule_based_claim_split(answer)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis setup failed: {exc}") from exc

    claim_results: List[ClaimResult] = []
    labels: List[str] = []
    evidence_scores: List[float] = []

    for c in claims:
        base = score_claim(req.domain.value, c)
        flags = detect_confidence_flags(c, base["evidence_score"])
        result = ClaimResult(
            claim=base["claim"],
            domain=req.domain,
            evidence_snippets=base["evidence_snippets"],
            label=base["label"],
            evidence_score=base["evidence_score"],
            confidence_flags=flags,
            nli_relation=base.get("nli_relation", "neutral"),
        )
        claim_results.append(result)
        labels.append(base["label"])
        evidence_scores.append(base["evidence_score"])

    hallucination_score, explanation, score_breakdown = aggregate_hallucination(labels, evidence_scores)
    analysis_details = {
        "answer_source": "pasted" if req.answer else "generated",
        "question": req.question,
        "answer": answer,
        "extracted_claims": claims,
        "matched_evidence_snippets": [c.evidence_snippets for c in claim_results],
        "score_breakdown": score_breakdown,
    }

    return AnalyzeResponse(
        domain=req.domain,
        question=req.question,
        answer=answer,
        claims=claim_results,
        hallucination_score=hallucination_score,
        explanation=explanation,
        score_breakdown=score_breakdown,
        analysis_details=analysis_details,
    )

