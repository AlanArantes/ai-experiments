"""
Bedrock Knowledge Base Retriever module for SRE Runbooks.
Queries Amazon Bedrock Knowledge Bases using bedrock-agent-runtime.
Includes an intelligent local Markdown retriever fallback for offline testing.
"""

import os
import sys
import glob
from pathlib import Path
from typing import List, Dict, Any
import boto3

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

class SREKnowledgeBaseRetriever:
    """
    Retrieves relevant SRE runbooks, postmortems, and SOP guidelines from
    Amazon Bedrock Knowledge Base.
    """

    def __init__(self, kb_id: str = None, region: str = "us-east-1"):
        self.kb_id = kb_id or os.environ.get("BEDROCK_KB_ID", "")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.local_runbooks_dir = Path(__file__).parent / "runbooks"
        
        try:
            self.agent_runtime = boto3.client("bedrock-agent-runtime", region_name=self.region)
        except Exception:
            self.agent_runtime = None

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Executes semantic retrieval. If AWS Bedrock KB is configured, calls AWS API;
        otherwise falls back to local runbook indexing.
        """
        if self.kb_id and self.agent_runtime:
            try:
                response = self.agent_runtime.retrieve(
                    knowledgeBaseId=self.kb_id,
                    retrievalQuery={"text": query},
                    retrievalConfiguration={
                        "vectorSearchConfiguration": {
                            "numberOfResults": top_k
                        }
                    }
                )
                results = []
                for item in response.get("retrievalResults", []):
                    content = item.get("content", {}).get("text", "")
                    uri = item.get("location", {}).get("s3Location", {}).get("uri", "s3://bedrock-kb-sre/")
                    score = item.get("score", 1.0)
                    results.append({
                        "content": content,
                        "source": uri,
                        "score": score
                    })
                if results:
                    return results
            except Exception as e:
                print(f"[!] Bedrock Agent Runtime call failed ({e}). Falling back to local runbook store.")

        return self._local_runbook_search(query, top_k=top_k)

    def _local_runbook_search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Local semantic/keyword scoring across markdown runbooks.
        """
        query_words = set(query.lower().replace("-", " ").replace("_", " ").split())
        scored_docs = []

        if not self.local_runbooks_dir.exists():
            return []

        for md_file in self.local_runbooks_dir.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
                text_lower = text.lower()
                
                # Simple BM25-like overlap score for offline verification
                score = 0
                for word in query_words:
                    if len(word) > 2 and word in text_lower:
                        score += text_lower.count(word)

                if score > 0:
                    scored_docs.append({
                        "content": text.strip(),
                        "source": f"file:///{md_file.as_posix()}",
                        "score": float(score)
                    })
            except Exception:
                continue

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
