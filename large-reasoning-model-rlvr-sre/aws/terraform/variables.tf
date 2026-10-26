variable "aws_region" {
  description = "AWS Region to deploy infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project identifier for resource naming and tagging"
  type        = string
  default     = "lrm-sre-rlvr"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "eks_cluster_name" {
  description = "Name of the EKS cluster for the RLVR sandbox"
  type        = string
  default     = "lrm-sre-sandbox-eks"
}

variable "sagemaker_instance_type" {
  description = "EC2 instance type for SageMaker Training"
  type        = string
  default     = "ml.g5.12xlarge"
}
