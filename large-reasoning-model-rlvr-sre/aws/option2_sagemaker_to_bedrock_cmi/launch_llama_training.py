"""
Option 2: Launcher for SageMaker Meta-Llama Training.
Submits the training job with Llama-3.1-8B, outputting to S3 ready for Bedrock CMI.
"""

import sys
import argparse

def launch_llama(
    role_arn: str,
    instance_type: str = "ml.g5.12xlarge",
    model_id: str = "meta-llama/Llama-3.1-8B",
    hf_token: str = None,
    spot: bool = True
):
    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        print("[!] Error: 'sagemaker' package not installed. Run: uv add sagemaker boto3")
        sys.exit(1)

    print("=" * 70)
    print(" [Option 2] Launching SageMaker Llama Training for Bedrock CMI")
    print(f" Target Model: {model_id} | Instance: {instance_type} | Spot: {spot}")
    print("=" * 70)

    env_vars = {}
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    estimator = PyTorch(
        entry_point="aws/option2_sagemaker_to_bedrock_cmi/train_llama_sagemaker.py",
        source_dir=".",
        role=role_arn,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type=instance_type,
        use_spot_instances=spot,
        max_run=3600 * 4 if spot else 3600 * 2,
        max_wait=3600 * 4 if spot else None,
        hyperparameters={
            "model-id": model_id,
            "group-size": 4,
            "lr": 1e-5
        },
        environment=env_vars
    )

    print("\n[*] Submitting SageMaker Llama Training Job...")
    estimator.fit()

    print("\n[OK] Training completed!")
    print(f"[*] S3 Model Artifacts for Bedrock Import: {estimator.model_data}")
    print("\n[*] Next Step: Import to Bedrock using:")
    print(f"    uv run python aws/option2_sagemaker_to_bedrock_cmi/import_to_bedrock_cmi.py --model-s3-uri '{estimator.model_data}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker Llama RLVR Training for Bedrock CMI")
    parser.add_argument("--role-arn", type=str, required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--instance-type", type=str, default="ml.g5.12xlarge")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face access token for Meta-Llama gating")
    parser.add_argument("--spot", action="store_true", default=True)
    args = parser.parse_args()

    launch_llama(
        role_arn=args.role_arn,
        instance_type=args.instance_type,
        model_id=args.model_id,
        hf_token=args.hf_token,
        spot=args.spot
    )
