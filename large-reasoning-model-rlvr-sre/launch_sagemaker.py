"""
SageMaker Launcher Script:
Submits a distributed GPU training job to AWS SageMaker using the PyTorch Estimator.

Prerequisites:
    uv add sagemaker boto3
    aws configure

Run via:
    uv run python launch_sagemaker.py --role-arn "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
"""

import argparse
import sys

def launch(role_arn: str, instance_type: str = "ml.g5.12xlarge", s3_data_uri: str = None, use_spot: bool = False):
    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        print("[!] Error: 'sagemaker' package is not installed.")
        print("    Install it via: uv add sagemaker boto3")
        sys.exit(1)

    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"[*] SageMaker Session initialized. Default S3 Bucket: s3://{bucket}")

    # Hyperparameters passed to train_sagemaker.py
    hyperparameters = {
        "model-id": "Qwen/Qwen2.5-Coder-7B",
        "group-size": 4,
        "seq-len": 512,
        "lr": 1e-5,
        "epsilon": 0.2,
        "beta-kl": 0.04,
        "epochs": 3
    }

    print(f"[*] Configuring PyTorch Estimator:")
    print(f"    - Instance Type: {instance_type}")
    print(f"    - Entry Point: train_sagemaker.py")
    print(f"    - Spot Instances: {use_spot}")

    estimator = PyTorch(
        entry_point="train_sagemaker.py",
        source_dir=".",                        # Packages current folder & requirements.txt
        role=role_arn,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=1,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        use_spot_instances=use_spot,
        max_run=3600 * 4 if use_spot else 3600 * 2,
        max_wait=3600 * 4 if use_spot else None,
        environment={
            # Add HF_TOKEN if downloading gated models like Llama-3
            # "HF_TOKEN": "your_token"
        }
    )

    inputs = {}
    if s3_data_uri:
        print(f"[*] Connecting input training data: {s3_data_uri}")
        inputs["training"] = s3_data_uri

    print("\n[*] Submitting Training Job to AWS SageMaker...")
    print("    SageMaker will provision the GPU instance, pull the container, run training,")
    print("    and sync the final model to S3 automatically.\n")
    estimator.fit(inputs=inputs)

    print("\n[OK] Training completed!")
    print(f"[*] Trained Model Artifacts location: {estimator.model_data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit LRM Training Job to AWS SageMaker")
    parser.add_argument("--role-arn", type=str, required=True, help="AWS IAM Role ARN for SageMaker execution")
    parser.add_argument("--instance-type", type=str, default="ml.g5.12xlarge", help="EC2 GPU instance type (e.g. ml.g5.2xlarge, ml.g5.12xlarge, ml.p4d.24xlarge)")
    parser.add_argument("--s3-data-uri", type=str, default=None, help="S3 URI for training dataset (e.g. s3://my-bucket/sre-traces/)")
    parser.add_argument("--spot", action="store_true", help="Use Managed Spot Instances to reduce GPU compute costs by up to 70%")
    args = parser.parse_args()

    launch(role_arn=args.role_arn, instance_type=args.instance_type, s3_data_uri=args.s3_data_uri, use_spot=args.spot)
