# -----------------------------------------------------------------------------
# Autonomous Healing: CloudWatch Alarm -> EventBridge -> AWS Lambda Orchestrator
# -----------------------------------------------------------------------------

resource "aws_iam_role" "lambda_sre_role" {
  name = "${var.project_name}-lambda-sre-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_sre_role.name
}

resource "aws_iam_role_policy_attachment" "lambda_vpc_access" {
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
  role       = aws_iam_role.lambda_sre_role.name
}

# Policy allowing Lambda to invoke SageMaker Endpoint and describe EKS
resource "aws_iam_policy" "lambda_sre_policy" {
  name        = "${var.project_name}-lambda-sre-policy"
  description = "Allows Lambda to query SageMaker reasoning endpoints and remediate on EKS"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint",
          "sagemaker:InvokeEndpointAsync"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "eks:DescribeCluster",
          "eks:AccessKubernetesApi"
        ]
        Resource = aws_eks_cluster.sandbox_eks.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_sre_attach" {
  policy_arn = aws_iam_policy.lambda_sre_policy.arn
  role       = aws_iam_role.lambda_sre_role.name
}

# -----------------------------------------------------------------------------
# CloudWatch Alarm: Simulating Pod Crash or Error Spike
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_metric_alarm" "pod_crash_alarm" {
  alarm_name          = "${var.project_name}-pod-crash-detected"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "pod_restart_rate"
  namespace           = "ContainerInsights"
  period              = 60
  statistic           = "Sum"
  threshold           = 1
  alarm_description   = "Triggers autonomous LRM remediation when pod restarts or crashes are detected"
  treat_missing_data  = "notBreaching"
}

# -----------------------------------------------------------------------------
# EventBridge Rule: Dispatches CloudWatch Alarm to Lambda
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_event_rule" "sre_alarm_rule" {
  name        = "${var.project_name}-sre-alarm-trigger"
  description = "Routes cluster degraded alarms to the LRM Lambda orchestrator"

  event_pattern = jsonencode({
    source      = ["aws.cloudwatch"]
    detail-type = ["CloudWatch Alarm State Change"]
    detail = {
      alarmName = [aws_cloudwatch_metric_alarm.pod_crash_alarm.alarm_name]
      state = {
        value = ["ALARM"]
      }
    }
  })
}

# Security group for Lambda inside the VPC
resource "aws_security_group" "lambda_sg" {
  name        = "${var.project_name}-lambda-sg"
  description = "Security group for SRE Lambda Orchestrator"
  vpc_id      = aws_vpc.lrm_vpc.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-lambda-sg" }
}
