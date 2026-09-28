"""judge_evaluator.py
Semantic Layer: Deterministic LLM-as-a-Judge Implementation with Closed-World Assumption
From: 'Fortifying Agentic AI: Building Resilient Test Scaffolding and Defense-in-Depth for LLM Systems'
by Alan Arantes
"""

import os
import re
from typing import Optional
from openai import OpenAI
from eval_models import AtomicClaim, ClaimStatus, FaithfulnessEvaluation, Verdict

SYSTEM_AUDIT_PROMPT = """You are a rigorous factual integrity auditor for RAG systems. Evaluate whether the Generated Answer contains ONLY facts provable by the Retrieved Context. Follow the Closed-World Assumption. Any external fact or presumption is an ungrounded hallucination.
1. Deconstruct the answer into atomic claims.
2. Locate verbatim evidence in the context for each claim.
3. Mark as UNSUPPORTED if absent.
4. If >= 1 claim is UNSUPPORTED, verdict = FAIL.
Always provide full reasoning prior to computing the verdict."""


def _offline_closed_world_audit(query: str, context: str, answer: str) -> FaithfulnessEvaluation:
    """Offline deterministic Closed-World evaluator when OpenAI API is not active.
    
    Strictly verifies each sentence/claim against the context to prevent hallucination.
    """
    # Deconstruct answer into atomic claims (sentences and clauses)
    raw_clauses = re.split(r"(?:[.!?]\s+|;\s*|\s+and\s+(?:also\s+|additionally\s+)?)", answer)
    raw_claims = [c.strip() for c in raw_clauses if len(c.strip()) > 5]
    if not raw_claims:
        raw_claims = [answer.strip()]

    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
        "by", "about", "against", "between", "into", "through", "during", "before",
        "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
        "could", "shall", "should", "will", "would", "may", "might", "must", "all",
        "also", "any", "both", "each", "few", "more", "most", "other", "some", "such"
    }

    atomic_claims = []
    hallucinated_facts = []
    norm_context = re.sub(r"\s+", " ", context).lower()

    for claim in raw_claims:
        norm_claim = re.sub(r"\s+", " ", claim).lower()
        words = [w for w in re.findall(r"\b[a-z]{3,}\b", norm_claim) if w not in stopwords]
        
        # Check presence of content words
        unsupported_words = [w for w in words if w not in norm_context]
        is_supported = (len(unsupported_words) == 0 and len(words) > 0)
        
        evidence_quote = ""
        if is_supported:
            # Locate an evidence window in context
            for w in words:
                idx = norm_context.find(w)
                if idx != -1:
                    snippet_start = max(0, idx - 15)
                    snippet_end = min(len(context), idx + len(w) + 40)
                    evidence_quote = context[snippet_start:snippet_end].strip()
                    break

            atomic_claims.append(
                AtomicClaim(
                    claim=claim,
                    evidence_quote=evidence_quote or "Grounded in retrieved context",
                    status=ClaimStatus.SUPPORTED
                )
            )
        else:
            atomic_claims.append(
                AtomicClaim(
                    claim=claim,
                    evidence_quote="",
                    status=ClaimStatus.UNSUPPORTED
                )
            )
            hallucinated_facts.append(
                f"{claim} (Unsupported tokens: {', '.join(unsupported_words) if unsupported_words else 'unverified'})"
            )

    total_claims = len(atomic_claims)
    supported_claims = sum(1 for c in atomic_claims if c.status == ClaimStatus.SUPPORTED)
    faithfulness_score = (supported_claims / total_claims) if total_claims > 0 else 0.0
    verdict = Verdict.PASS if (supported_claims == total_claims and total_claims > 0) else Verdict.FAIL

    reasoning = (
        f"Analytical Audit (Closed-World Assumption): Deconstructed answer into {total_claims} atomic claims. "
        f"Found {supported_claims} supported by context and {len(hallucinated_facts)} unsupported claims. "
        f"Score: {faithfulness_score:.2f}. Verdict: {verdict.value}."
    )

    return FaithfulnessEvaluation(
        atomic_claims=atomic_claims,
        reasoning=reasoning,
        hallucinated_facts=hallucinated_facts,
        faithfulness_score=faithfulness_score,
        verdict=verdict
    )


def evaluate_faithfulness(
    query: str,
    context: str,
    answer: str,
    api_key: Optional[str] = None,
    client: Optional[OpenAI] = None,
    force_offline: bool = False
) -> FaithfulnessEvaluation:
    """Evaluates factual faithfulness using OpenAI Structured Outputs or deterministic offline auditor.
    
    Enforces Closed-World Assumption with zero temperature to eradicate evaluation drift.
    """
    effective_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not force_offline and (client is not None or effective_key):
        try:
            if client is None:
                client = OpenAI(api_key=effective_key)

            prompt = f"[QUERY]: {query}\n[CONTEXT]: {context}\n[ANSWER]: {answer}"
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_AUDIT_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format=FaithfulnessEvaluation
            )
            parsed: Optional[FaithfulnessEvaluation] = response.choices[0].message.parsed
            if parsed:
                return parsed
        except Exception as e:
            # Fall back to offline deterministic auditor if API call encounters network/auth issues
            print(f"[Notice] LLM API call error ({e}). Falling back to deterministic auditor.")

    # Offline Closed-World Deterministic Evaluation
    return _offline_closed_world_audit(query, context, answer)
