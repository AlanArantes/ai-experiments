# ==============================================================================
# Option 3 (Option 1 + RAG): Bedrock Serverless SRE Agent + Bedrock Knowledge Base
# Zero GPU Endpoints - Fully Event-Driven - Grounded in Internal Runbooks
# ==============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.50"
    }
  }
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "bedrock_model_id" {
  type        = string
  default     = "anthropic.claude-3-5-haiku-20241022-v1:0"
  description = "Bedrock Foundation Model ID for autonomous SRE reasoning"
}

# ------------------------------------------------------------------------------
# 1. S3 Bucket for SRE Runbooks & Postmortems
# ------------------------------------------------------------------------------
resource "aws_s3_bucket" "sre_runbooks" {
  bucket        = "lrm-sre-runbooks-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "sre_runbooks_versioning" {
  bucket = aws_s3_bucket.sre_runbooks.id
  versioning_configuration {
    status = "Enabled"
  }
}

data "aws_caller_identity" "current" {}

# Upload sample runbooks
resource "aws_s3_object" "runbook_oom" {
  bucket = aws_s3_bucket.sre_runbooks.id
  key    = "runbooks/runbook_oom_killed.md"
  source = "${path.module}/runbooks/runbook_oom_killed.md"
  etag   = filemd5("${path.module}/runbooks/runbook_oom_killed.md")
}

resource "aws_s3_object" "runbook_502" {
  bucket = aws_s3_bucket.sre_runbooks.id
  key    = "runbooks/runbook_502_bad_gateway.md"
  source = "${path.module}/runbooks/runbook_502_bad_gateway.md"
  etag   = filemd5("${path.module}/runbooks/runbook_502_bad_gateway.md")
}

# ------------------------------------------------------------------------------
# 2. IAM Role for Bedrock Knowledge Base
# ------------------------------------------------------------------------------
resource "aws_iam_role" "bedrock_kb_role" {
  name = "lrm-bedrock-kb-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
        }
      }
    ]
  })
}

resource "aws_iam_policy" "bedrock_kb_s3_policy" {
  name = "lrm-bedrock-kb-s3-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.sre_runbooks.arn,
          "${aws_s3_bucket.sre_runbooks.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "kb_s3_attach" {
  role       = aws_iam_role.bedrock_kb_role.name
  policy_arn = aws_iam_policy.bedrock_kb_s3_policy.arn
}

# ------------------------------------------------------------------------------
# 3. Bedrock Knowledge Base Resource
# ------------------------------------------------------------------------------
resource "aws_bedrockagent_knowledge_base" "sre_kb" {
  name     = "lrm-sre-runbooks-kb"
  role_arn = aws_iam_role.bedrock_kb_role.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = "arn:aws:aoss:${var.aws_region}:${data.aws_caller_identity.current.account_id}:collection/sre-kb-collection"
      vector_index_name = "sre-runbooks-index"
      field_mapping {
        vector_field   = "vector"
        text_field     = "text"
        metadata_field = "metadata"
      }
    }
  }
}

resource "aws_bedrockagent_data_source" "sre_kb_s3_source" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.sre_kb.id
  name              = "sre-runbooks-s3-datasource"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = aws_s3_bucket.sre_runbooks.arn
      inclusion_prefixes = ["runbooks/"]
    }
  }
}

# ------------------------------------------------------------------------------
# 4. Lambda SRE Orchestrator with RAG Grounding
# ------------------------------------------------------------------------------
resource "aws_iam_role" "lambda_rag_role" {
  name = "lrm-sre-bedrock-rag-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_policy" "lambda_rag_policy" {
  name = "lrm-sre-bedrock-rag-policy"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:Converse",
          "bedrock:InvokeModel",
          "bedrock-agent-runtime:Retrieve"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_rag_attach" {
  role       = aws_iam_role.lambda_rag_role.name
  policy_arn = aws_iam_policy.lambda_rag_policy.arn
}

resource "aws_lambda_function" "bedrock_rag_orchestrator" {
  function_name = "lrm-sre-bedrock-rag-orchestrator"
  role          = aws_iam_role.lambda_rag_role.arn
  handler       = "orchestrator_bedrock_rag.lambda_handler"
  runtime       = "python3.11"
  timeout       = 120
  memory_size   = 512

  filename         = "${path.module}/lambda_payload.zip"
  source_code_hash = filebase64sha256("${path.module}/orchestrator_bedrock_rag.py")

  environment {
    variables = {
      BEDROCK_MODEL_ID = var.bedrock_model_id
      BEDROCK_KB_ID    = aws_bedrockagent_knowledge_base.sre_kb.id
      AWS_REGION       = var.aws_region
    }
  }
}

output "bedrock_kb_id" {
  description = "Bedrock Knowledge Base ID"
  value       = aws_bedrockagent_knowledge_base.sre_kb.id
}

output "lambda_rag_arn" {
  description = "ARN of the RAG-grounded SRE Lambda Orchestrator"
  value       = aws_lambda_function.bedrock_rag_orchestrator.arn
}
