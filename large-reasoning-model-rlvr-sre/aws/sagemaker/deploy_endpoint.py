"""
SageMaker Endpoint Deployment Script:
Deploys the fine-tuned Large Reasoning Model to an AWS SageMaker Real-Time Endpoint
using the high-throughput vLLM Deep Learning Container (DLC) on an ml.g5.2xlarge instance.
"""

import sys
import argparse
import boto3
import json

def deploy(model_s3_uri: str, role_arn: str, endpoint_name: str = "lrm-sre-reasoning-endpoint", instance_type: str = "ml.g5.2xlarge"):
    try:
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel
    except ImportError:
        print("[!] Error: 'sagemaker' package not installed. Run: uv add sagemaker boto3")
        sys.exit(1)

    print("=" * 70)
    print(" AWS SageMaker Real-Time Endpoint Deployment (vLLM)")
    print("=" * 70)
    print(f"[*] Model S3 URI: {model_s3_uri}")
    print(f"[*] Endpoint Name: {endpoint_name}")
    print(f"[*] Instance Type: {instance_type}")
    print(f"[*] Execution Role: {role_arn}")

    # Configuration for vLLM container
    hub = {
        "HF_MODEL_ID": "/opt/ml/model",
        "SM_NUM_GPUS": "1",
        "DTYPE": "bfloat16",
        "MAX_MODEL_LEN": "4096",
        "TENSOR_PARALLEL_DEGREE": "1"
    }

    # Retrieve official Hugging Face TGI / vLLM DLC image URI
    from sagemaker.huggingface import get_huggingface_llm_image_uri
    image_uri = get_huggingface_llm_image_uri(
        backend="huggingface",  # or "vllm"
        version="2.3.1"
    )
    print(f"[*] Using AWS Deep Learning Container image: {image_uri}")

    # Create HuggingFaceModel
    model = HuggingFaceModel(
        model_data=model_s3_uri,
        role=role_arn,
        image_uri=image_uri,
        env=hub
    )

    print("\n[*] Deploying endpoint (this provisions the GPU instance in ~4-6 minutes)...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    print(f"\n[✓] Endpoint '{endpoint_name}' is LIVE!")

    # Test the deployed reasoning model
    print("[*] Testing live reasoning inference...")
    test_payload = {
        "inputs": "<context>ALERT: ServiceDegraded. Pod payment-processor in CrashLoopBackOff. Exit Code 137 OOMKilled.</context>\n<query>Diagnose and provide remediation action.</query>",
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.2
        }
    }
    
    response = predictor.predict(test_payload)
    print(f"[✓] Model Response Preview:\n{json.dumps(response, indent=2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy tuned LRM to SageMaker Real-Time Endpoint")
    parser.add_argument("--model-s3-uri", type=str, required=True, help="S3 URI containing model.tar.gz (e.g. s3://bucket/model.tar.gz)")
    parser.add_argument("--role-arn", type=str, required=True, help="IAM Role ARN for SageMaker")
    parser.add_argument("--endpoint-name", type=str, default="lrm-sre-reasoning-endpoint")
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge")
    args = parser.parse_args()

    deploy(
        model_s3_uri=args.model_s3_uri,
        role_arn=args.role_arn,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type
    )
