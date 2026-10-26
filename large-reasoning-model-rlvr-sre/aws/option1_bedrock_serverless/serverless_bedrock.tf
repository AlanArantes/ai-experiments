# -----------------------------------------------------------------------------
# Option 1: Pure Bedrock Serverless Autonomous SRE Infrastructure
# Provisions Lambda + EventBridge + Bedrock IAM (Zero SageMaker GPU costs)
# -----------------------------------------------------------------------------

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "lrm-bedrock-serverless-sre"
}

variable "bedrock_model_id" {
  description = "Bedrock foundation model for reasoning (Claude 3.5 Haiku or Llama 3.3 70B)"
  type        = string
  default     = "anthropic.claude-3-5-haiku-20241022-v1:0"
}

# 1. IAM Role for Serverless SRE Lambda
resource "aws_iam_role" "bedrock_sre_lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.bedrock_sre_lambda_role.name
}

resource "aws_iam_policy" "bedrock_sre_policy" {
  name        = "${var.project_name}-bedrock-policy"
  description = "Allows Lambda to invoke Amazon Bedrock foundation models"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = [
          "eks:DescribeCluster",
          "eks:AccessKubernetesApi"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "bedrock_sre_attach" {
  policy_arn = aws_iam_policy.bedrock_sre_policy.arn
  role       = aws_iam_role.bedrock_sre_lambda_role.name
}

# 2. Package & Deploy Lambda Function
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/orchestrator_bedrock.py"
  output_path = "${path.module}/lambda_payload.zip"
}

resource "aws_lambda_function" "bedrock_sre_lambda" {
  function_name    = "${var.project_name}-orchestrator"
  role             = aws_iam_role.bedrock_sre_lambda_role.arn
  handler          = "orchestrator_bedrock.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 60
  memory_size      = 256

  environment {
    variables = {
      BEDROCK_MODEL_ID = var.bedrock_model_id
      AWS_REGION       = var.aws_region
    }
  }
}

# 3. EventBridge Rule triggering Lambda on CloudWatch Alarm
resource "aws_cloudwatch_event_rule" "alarm_rule" {
  name        = "${var.project_name}-trigger-rule"
  description = "Captures cluster alarms and triggers the serverless Bedrock reasoning agent"

  event_pattern = jsonencode({
    source      = ["aws.cloudwatch"]
    detail-type = ["CloudWatch Alarm State Change"]
    detail = {
      state = { value = ["ALARM"] }
    }
  })
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.alarm_rule.name
  target_id = "TriggerBedrockSRELambda"
  arn       = aws_lambda_function.bedrock_sre_lambda.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.bedrock_sre_lambda.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.alarm_rule.arn
}

output "lambda_function_arn" {
  value = aws_lambda_function.bedrock_sre_lambda.arn
}
