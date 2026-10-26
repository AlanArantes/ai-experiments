"""
Option 2: Invoke the Custom-Imported Model in Amazon Bedrock.
Demonstrates querying the fine-tuned reasoning model using standard Bedrock Converse API
with its private Bedrock Model ARN.
"""

import sys
import json
import argparse
import boto3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def test_inference(model_arn: str, region: str = "us-east-1"):
    print("=" * 70)
    print(" [Option 2] Querying Custom LRM via Amazon Bedrock Converse API")
    print(f" Custom Model ARN: {model_arn} | Region: {region}")
    print("=" * 70)

    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

    prompt = (
        "<context>\n"
        "ALERT: ServiceDegraded in namespace sandbox-app.\n"
        "Pod catalog-api-4b2z is healthy, but Ingress returns 502 Bad Gateway.\n"
        "Service selector is 'app: catalog-api', but pod labels are 'app.kubernetes.io/name: catalog-api'.\n"
        "</context>\n"
        "Diagnose inside <think>, propose kubectl fix inside <action>, and verdict inside <verdict>."
    )

    print("\n[*] Sending prompt to Custom Bedrock Model...")
    try:
        response = bedrock_runtime.converse(
            modelId=model_arn,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.1, "maxTokens": 512}
        )
        output_text = response["output"]["message"]["content"][0]["text"]
        print("\n[OK] Model Response Received:")
        print(output_text)
    except Exception as e:
        print(f"[!] Warning / Offline simulation: {e}")
        simulated_output = (
            "<think>\n"
            "Pod is running, but Ingress returns 502. Checking Service endpoints.\n"
            "Endpoints are empty because Service selector 'app: catalog-api' does not match pod labels 'app.kubernetes.io/name: catalog-api'.\n"
            "Remediation: Patch Service selector to match pod labels.\n"
            "</think>\n"
            "<action>\n"
            "{\"tool\": \"kubectl\", \"cmd\": \"patch service catalog-service -n sandbox-app --patch '{\\\"spec\\\":{\\\"selector\\\":{\\\"app.kubernetes.io/name\\\":\\\"catalog-api\\\"}}}'\"}\n"
            "</action>\n"
            "<verdict>\n"
            "{\"root_cause\": \"SELECTOR_LABEL_TYPO\", \"status\": \"RESOLVED\"}\n"
            "</verdict>"
        )
        print("\n[OK] Simulated Output (for local offline verification):")
        print(simulated_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Custom Bedrock Model")
    parser.add_argument("--model-arn", type=str, default="arn:aws:bedrock:us-east-1:123456789012:imported-model/lrm-sre-reasoning-llama3-1-8b")
    parser.add_argument("--region", type=str, default="us-east-1")
    args = parser.parse_args()

    test_inference(model_arn=args.model_arn, region=args.region)
