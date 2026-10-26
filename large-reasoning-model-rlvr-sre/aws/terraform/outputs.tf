output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.lrm_vpc.id
}

output "private_subnets" {
  description = "Private Subnet IDs for SageMaker and EKS"
  value       = aws_subnet.private[*].id
}

output "sagemaker_security_group_id" {
  description = "Security Group ID for SageMaker instances"
  value       = aws_security_group.sagemaker_sg.id
}

output "sagemaker_execution_role_arn" {
  description = "IAM Role ARN for SageMaker Training and Endpoints"
  value       = aws_iam_role.sagemaker_execution_role.arn
}

output "eks_cluster_name" {
  description = "EKS Cluster Name"
  value       = aws_eks_cluster.sandbox_eks.name
}

output "eks_cluster_endpoint" {
  description = "EKS Cluster API Endpoint"
  value       = aws_eks_cluster.sandbox_eks.endpoint
}

output "data_lake_bucket" {
  description = "S3 Data Lake Bucket Name"
  value       = aws_s3_bucket.data_lake.bucket
}

output "models_bucket" {
  description = "S3 Model Checkpoints Bucket Name"
  value       = aws_s3_bucket.models_lake.bucket
}
