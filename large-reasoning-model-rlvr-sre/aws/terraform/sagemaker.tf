# -----------------------------------------------------------------------------
# Amazon SageMaker: Execution Role, IAM Policies, and VPC Endpoints
# -----------------------------------------------------------------------------

resource "aws_iam_role" "sagemaker_execution_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
  role       = aws_iam_role.sagemaker_execution_role.name
}

# Fine-grained access policy for S3 buckets, Bedrock, and EKS describe
resource "aws_iam_policy" "sagemaker_lrm_custom_policy" {
  name        = "${var.project_name}-sagemaker-lrm-policy"
  description = "Allows SageMaker to access project S3 buckets, call Bedrock, and query EKS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.data_lake.arn,
          "${aws_s3_bucket.data_lake.arn}/*",
          aws_s3_bucket.models_lake.arn,
          "${aws_s3_bucket.models_lake.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:CreateBatchInferenceJob",
          "bedrock:GetBatchInferenceJob"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:ListClusters",
          "eks:AccessKubernetesApi"
        ]
        Resource = aws_eks_cluster.sandbox_eks.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_lrm_attach" {
  policy_arn = aws_iam_policy.sagemaker_lrm_custom_policy.arn
  role       = aws_iam_role.sagemaker_execution_role.name
}

# -----------------------------------------------------------------------------
# VPC S3 Gateway Endpoint (Zero data transfer fees to S3)
# -----------------------------------------------------------------------------
resource "aws_vpc_endpoint" "s3_endpoint" {
  vpc_id            = aws_vpc.lrm_vpc.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]

  tags = { Name = "${var.project_name}-s3-endpoint" }
}
