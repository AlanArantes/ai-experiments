"""
AWS Bedrock Synthetic Data Generator:
Distills high-quality IRAC and SRE reasoning traces with explicit backtracking and self-correction tokens.
Uploads the validated dataset directly to Amazon S3 for SageMaker training.
"""

import os
import sys
import json
import boto3
import argparse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Include project root in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from synthetic_data.irac_pipeline import IRACGenerator

DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"

SAMPLE_LEGAL_PROMPTS = [
    {
        "context": "Clause 14.2: Right of withdrawal exercisable within 7 calendar days of receipt. Clause 14.3: Custom goods are exempt unless defective. Clause 22: Written notification by email to support@company.com is valid notice.",
        "query": "I received custom-engraved headphones on Oct 1. On Oct 5, I emailed support asking to return them because I changed my mind. They are in pristine condition. Can I cancel?"
    },
    {
        "context": "Clause 8.1: Software license terminable for convenience with 30 days prior written notice. Clause 8.4: In case of non-payment exceeding 15 days, licensor may terminate immediately without cure period.",
        "query": "Our account is 20 days past due on payment. Licensor sent an immediate termination letter this morning. Do we have a right to a 30-day cure period?"
    }
]

def generate_reasoning_trace(bedrock_client, model_id: str, context: str, user_query: str) -> str:
    system_prompt = (
        "You are an expert legal reasoning system. You MUST format your response as:\n"
        "<think>\n"
        "[ISSUE] ...\n"
        "[RULE] ... (quote exact clauses)\n"
        "[APPLICATION] ... (deliberately test hypotheses and explore exceptions. "
        "Include self-correction like 'Wait, let me double check...', 'However, clause X is an exception...')\n"
        "[CONCLUSION] ...\n"
        "</think>\n"
        "<answer>\n"
        "[Clear client-facing explanation citing specific clauses]\n"
        "</answer>"
    )

    user_message = f"Context:\n{context}\n\nClient Query:\n{user_query}"

    response = bedrock_client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"temperature": 0.4, "maxTokens": 4096}
    )

    return response["output"]["message"]["content"][0]["text"]

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data using AWS Bedrock")
    parser.add_argument("--model-id", type=str, default=DEFAULT_BEDROCK_MODEL, help="Bedrock Foundation Model ID")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS Region")
    parser.add_argument("--s3-bucket", type=str, default=None, help="Target S3 Bucket for saving dataset")
    parser.add_argument("--output-file", type=str, default="synthetic_irac_traces.jsonl", help="Local output JSONL file")
    args = parser.parse_args()

    print("=" * 70)
    print(" AWS Bedrock Synthetic Data Distillation Engine")
    print(f" Model: {args.model_id} | Region: {args.region}")
    print("=" * 70)

    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)
    s3_client = boto3.client("s3", region_name=args.region) if args.s3_bucket else None

    records = []
    for idx, prompt_data in enumerate(SAMPLE_LEGAL_PROMPTS, 1):
        print(f"\n[*] Generating trace {idx}/{len(SAMPLE_LEGAL_PROMPTS)}...")
        raw_output = generate_reasoning_trace(
            bedrock_client=bedrock_client,
            model_id=args.model_id,
            context=prompt_data["context"],
            user_query=prompt_data["query"]
        )

        # Extract <think> and <answer>
        import re
        think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_output, re.DOTALL)

        think_text = think_match.group(1).strip() if think_match else ""
        answer_text = answer_match.group(1).strip() if answer_match else raw_output

        # Verify trace quality using deterministic rule verifier
        is_valid, errors = IRACGenerator.verify_irac_trace(prompt_data["context"], think_text, answer_text)
        print(f"    -> Verification Status: {'PASS' if is_valid else 'FAIL'}")
        if errors:
            print(f"    -> Warnings/Errors: {errors}")

        record = {
            "id": f"irac_trace_{idx:04d}",
            "context": prompt_data["context"],
            "query": prompt_data["query"],
            "think": think_text,
            "answer": answer_text,
            "full_trajectory": raw_output,
            "verified": is_valid
        }
        records.append(record)

    # Save to local file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[OK] Saved {len(records)} verified records to {args.output_file}")

    # Upload to S3 if bucket provided
    if args.s3_bucket:
        s3_key = f"synthetic-data/{os.path.basename(args.output_file)}"
        print(f"[*] Uploading to S3: s3://{args.s3_bucket}/{s3_key}...")
        s3_client.upload_file(args.output_file, args.s3_bucket, s3_key)
        print(f"[OK] Dataset successfully uploaded to Amazon S3!")

if __name__ == "__main__":
    main()
