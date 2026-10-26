"""
SageMaker Launcher for EKS RLVR Training:
Submits a distributed GPU training job connected to the Amazon EKS cluster over the private VPC.
"""

import sys
import argparse

def launch(
    role_arn: str,
    subnet_ids: list[str],
    security_group_ids: list[str],
    eks_cluster_name: str = "lrm-sre-sandbox-eks",
    instance_type: str = "ml.g5.12xlarge",
    instance_count: int = 1,
    s3_data_uri: str = None,
    spot: bool = True
):
    try:
        import sagemaker
        from sagemaker.pytorch import PyTorch
    except ImportError:
        print("[!] Error: 'sagemaker' package not installed. Run: uv add sagemaker boto3")
        sys.exit(1)

    print("=" * 70)
    print(" AWS SageMaker + EKS RLVR Training Launcher")
    print("=" * 70)
    print(f"[*] EKS Sandbox Cluster: {eks_cluster_name}")
    print(f"[*] Instance Type: {instance_type} (Count: {instance_count})")
    print(f"[*] Subnets: {subnet_ids}")
    print(f"[*] Security Groups: {security_group_ids}")
    print(f"[*] Managed Spot Instances: {spot}")

    hyperparameters = {
        "model-id": "Qwen/Qwen2.5-Coder-7B",
        "eks-cluster-name": eks_cluster_name,
        "group-size": 4,
        "seq-len": 512,
        "lr": 1e-5,
        "epochs": 2
    }

    estimator = PyTorch(
        entry_point="aws/sagemaker/train_eks_rlvr.py",
        source_dir=".",
        role=role_arn,
        framework_version="2.5.1",
        py_version="py311",
        instance_count=instance_count,
        instance_type=instance_type,
        subnets=subnet_ids,
        security_group_ids=security_group_ids,
        hyperparameters=hyperparameters,
        use_spot_instances=spot,
        max_run=3600 * 4 if spot else 3600 * 2,
        max_wait=3600 * 4 if spot else None
    )

    inputs = {}
    if s3_data_uri:
        inputs["training"] = s3_data_uri

    print("\n[*] Submitting job to AWS SageMaker...")
    estimator.fit(inputs=inputs)

    print("\n[OK] Training completed!")
    print(f"[*] Final S3 Model Artifacts: {estimator.model_data}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit SageMaker + EKS RLVR Training Job")
    parser.add_argument("--role-arn", type=str, required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--subnets", type=str, required=True, help="Comma-separated private subnet IDs")
    parser.add_argument("--security-groups", type=str, required=True, help="Comma-separated security group IDs")
    parser.add_argument("--eks-cluster", type=str, default="lrm-sre-sandbox-eks", help="Target EKS cluster name")
    parser.add_argument("--instance-type", type=str, default="ml.g5.12xlarge")
    parser.add_argument("--s3-data-uri", type=str, default=None)
    parser.add_argument("--spot", action="store_true", default=True)
    args = parser.parse_args()

    subnet_list = [s.strip() for s in args.subnets.split(",") if s.strip()]
    sg_list = [sg.strip() for sg in args.security_groups.split(",") if sg.strip()]

    launch(
        role_arn=args.role_arn,
        subnet_ids=subnet_list,
        security_group_ids=sg_list,
        eks_cluster_name=args.eks_cluster,
        instance_type=args.instance_type,
        s3_data_uri=args.s3_data_uri,
        spot=args.spot
    )
