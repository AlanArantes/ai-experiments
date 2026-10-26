# Building a Large Reasoning Model (LRM) with RLVR & Autonomous SRE

This repository contains the architecture, synthetic data pipelines, verification harnesses, and Reinforcement Learning with Verifiable Rewards (RLVR) implementations for training **Large Reasoning Models (LRMs)**. It demonstrates how to transition beyond autoregressive next-token prediction into **test-time compute scaling**, **explicit Chain-of-Thought (CoT) backtracking**, and **deterministic policy optimization using GRPO**.

Two domain specializations are implemented and analyzed:
1. **Legal Customer Service / Intake:** Structured **IRAC** (Issue, Rule, Application, Conclusion) reasoning traces with citation integrity and hypothesis backtracking.
2. **Autonomous SRE / Kubernetes Management:** A closed-loop RLVR environment where an agent diagnoses and remediates real cluster incidents (`OOMKilled`, `CrashLoopBackOff`, broken service selectors, missing ConfigMaps) against verifiable health probes.

---

## Architecture Overview

```
                      [ Pre-training Foundation ]
         (Decoder-only Transformer: RoPE, SwiGLU, RMSNorm, GQA)
          30%–50%+ Code, Math (LaTeX), Logic Proofs & Traces
                                  │
                                  ▼
                   [ Cold-Start SFT Phase ]
           Enforce <think>...</think> Structured Delimiters
           Synthetic Traces with Backtracking & Self-Correction
                                  │
                                  ▼
                   [ RLVR Post-Training (GRPO) ]
        Deterministic Rule Verifiers & Sandboxed Environments
        Group Relative Policy Optimization (Critic-Free VRAM)
                                  │
                                  ▼
                   [ Test-Time Compute Scaling ]
            Best-of-N Rollouts, MCTS & KV-Cache Caching
```

---

## The 5 Foundational Engineering Phases

### 1. Core Architecture & Pre-training Foundation

- **Model Architecture:** Standard decoder-only transformer with:
  - **RoPE (Rotary Position Embeddings)** with long context windows (32k–128k tokens) to support extensive multi-turn scratchpads.
  - **SwiGLU Activations & RMSNorm** for numerical stability under large batch sizes.
  - **Grouped-Query Attention (GQA)** to minimize KV-cache footprint during long token generation.
- **Pre-training Mixture:** 30% to 50%+ dedicated to code repositories, formal math, logic puzzles, algorithmic traces, and symbolic proofs to force the network to model long-range causal dependencies.

### 2. Supervised Fine-Tuning (SFT) for Reasoning Monologues

Reasoning requires the model to pause and formulate an internal monologue before emitting a final answer:
```xml
<think>
Step 1: Parse requirements...
Step 2: Check edge case where n = 0...
Wait, if n = 0, division by zero occurs. Let me re-evaluate...
Step 3: Corrected formula...
</think>
Final Answer: ...
```
- **Error Correction & Backtracking:** Training traces must contain self-correction tokens (`"Wait, that's not right..."`, `"Let me re-evaluate..."`). If a model only observes pristine linear solutions, it cannot recover from hallucinated branches during generation.

### 3. Post-Training: Reinforcement Learning with Verifiable Rewards (RLVR)

Unlike general RLHF that relies on neural reward models (which are prone to reward hacking), RLVR uses **deterministic, rule-based execution environments**:
- **Math:** Exact symbolic match via SymPy/parsers on canonicalized boxed answers.
- **Code & SRE:** Sandboxed unit tests, exit codes, and HTTP probe status codes.
- **Algorithm: GRPO (Group Relative Policy Optimization):** Eliminates the separate critic/value network required by PPO, significantly saving VRAM:

$$\mathcal{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E} \left[ \min\left( \frac{\pi_\theta}{\pi_{\text{old}}} \mathcal{A}_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) \mathcal{A}_i \right) \right] + \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$$

### 4. Test-Time Compute Scaling & Inference

- **Search at Test Time:** Best-of-N sampling, majority voting, or Monte Carlo Tree Search (MCTS) over intermediate reasoning thoughts.
- **Serving Efficiency:** Prefix caching (vLLM / SGLang) to reuse prompt KV caches across rollout candidates.

### 5. Open-Source Stack

- **Pre-training / SFT:** Megatron-LM, torchtune, Hugging Face TRL.
- **RLVR / GRPO:** OpenRLHF, verl, TRL.
- **Serving:** vLLM, SGLang.

---

## Domain Comparison

| Metric / Dimension | General Customer Service | Legal Customer Service (IRAC) | Autonomous SRE / Cloud |
| :--- | :--- | :--- | :--- |
| **Applicability** | Low (overkill & high latency) | **High** (high stakes, compliance) | **Ideal (Prime RLVR Target)** |
| **Ground Truth** | Subjective / Human preference | Verifiable statutory clauses & rules | **100% Deterministic (APIs, logs, probes)** |
| **Feedback Loop** | Neural reward model (unstable) | Rule engines & citation checkers | **Cluster state, exit codes, Prometheus** |
| **Thinking Need** | Low (standard transactional FAQ) | High (multi-clause exceptions) | High (diagnostic hypothesis testing) |

---

## Repository Structure

```
large-reasoning-model-rlvr-sre/
├── pyproject.toml                     # Project dependencies managed by uv
├── uv.toml                            # Native TLS configuration for uv
├── main.py                            # End-to-end verification and test harness
├── scenarios/
│   ├── scenario_definitions.py        # Scenario catalog, ground truths, and blast-radius rules
│   ├── oom_service.yaml               # K8s manifest: Memory leak with tight cgroup limits
│   ├── inject_oom.sh                  # Fault injection script for OOMKilled
│   ├── selector_mismatch.yaml         # K8s manifest: Catalog API with mismatched service selector
│   ├── inject_selector_mismatch.sh    # Fault injection script for selector mismatch
│   ├── missing_configmap.yaml         # K8s manifest: Deployment referencing missing ConfigMap key
│   └── inject_missing_configmap.sh    # Fault injection script for missing ConfigMap
├── verifier/
│   ├── verifier.sh                    # Bash verification script for pod invariants & HTTP probes
│   └── verifier.py                    # Cross-platform Python verifier module
├── rlvr/
│   ├── __init__.py
│   ├── k8s_env.py                     # Gymnasium KubernetesEnv interacting with ephemeral clusters
│   ├── trajectory_formatter.py        # SRETrajectoryFormatter: token encoding & loss masking
│   ├── reward_evaluator.py            # SRERewardEvaluator: multi-part composite scalar reward
│   └── grpo_loss.py                   # compute_grpo_loss: PyTorch GRPO policy loss implementation
├── synthetic_data/
│   ├── irac_pipeline.py               # 4-stage synthetic IRAC generator and citation checker
│   ├── sample_sre_trajectory.xml      # Full SRE episode (<context>, <think>, <action>, <observation>, <verdict>)
│   └── sample_irac_trajectory.xml     # Full Legal IRAC episode
└── podman/
    ├── run_k3s_sandbox.ps1            # Windows PowerShell launcher for ephemeral K3s via Podman
    ├── run_k3s_sandbox.sh             # Bash launcher for Linux / WSL
    ├── cleanup_sandbox.ps1            # Cleanup script for Podman containers
    └── podman-compose.k3s.yml         # Podman Compose definition for K3s
```

---

## Quickstart & Verification with `uv`

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible Python environment management.

### 1. Install & Sync Dependencies
```powershell
uv sync
```

### 2. Run the End-to-End Test Suite
Executes scenario validation, trajectory formatting, reward evaluation, GRPO loss computation, and IRAC trace verification:
```powershell
uv run python main.py
```

Expected output:
```text
======================================================================
 Large Reasoning Model (LRM) - RLVR & Trajectory Evaluation
======================================================================

[1] Loaded 3 SRE RLVR Scenarios:
  • oom_misconfigured_limits: App pod crashing under moderate load... (Root Cause: MEMORY_LIMIT_TOO_LOW)
  • service_selector_mismatch: Ingress reports 502 Bad Gateway... (Root Cause: SELECTOR_LABEL_TYPO)
  • missing_configmap_key: Auth service pod fails to start... (Root Cause: MISSING_CONFIGMAP_KEY)

[2] Testing SRERewardEvaluator with Sample SRE Trajectory...
  -> Total Reward Score: 1.440
  -> Breakdown: {'r_health': 1.0, 'r_diagnosis': 0.5, 'r_penalty': 0.0, 'r_efficiency': 0.06}

[3] Testing SRETrajectoryFormatter & Loss Masking...
  -> Input IDs shape: torch.Size([64])
  -> Total sequence tokens: 15, Agent tokens (masked for loss): 6

[4] Testing Group Relative Policy Optimization (GRPO) Loss...
  -> GRPO Loss computed across 4 rollouts: 0.4838

[5] Testing Legal IRAC Synthetic Data Validator...
  -> IRAC Trace Valid: True
  -> Passed all structural, citation, and backtracking checks!

======================================================================
 [OK] All verification checks completed successfully.
======================================================================
```

---

## Running Ephemeral Kubernetes Sandboxes with Podman

To run live rollout episodes during RLVR training, you can spin up dedicated, single-node K3s clusters inside Podman in seconds.

### Launch K3s via PowerShell (Windows)
```powershell
.\podman\run_k3s_sandbox.ps1 -ClusterName "k8s-sandbox-01" -ApiPort 6443
```

### Launch K3s via Bash (WSL / Linux)
```bash
chmod +x podman/run_k3s_sandbox.sh
./podman/run_k3s_sandbox.sh "k8s-sandbox-01" 6443
```

### Verify Cluster Connection
```powershell
kubectl --kubeconfig="./k8s-output/kubeconfig.yaml" get nodes
```

### Clean Up Sandbox
```powershell
.\podman\cleanup_sandbox.ps1 -ClusterName "k8s-sandbox-01"
```

---

## SRE RLVR Reward Formulation

The composite reward $R \in [-2.0, 1.5]$ evaluates four key dimensions:
1. **Health Verification Probe ($R_{\text{health}}$):** $+1.0$ if the HTTP probe and pod invariants pass 3 consecutive checks; $-0.5$ if the episode ends unhealthy.
2. **Diagnostic Root Cause Verdict ($R_{\text{diagnosis}}$):** $+0.5$ for exact string match on ground truth root cause inside `<verdict>`, $-0.2$ for mismatch, $-0.3$ for malformed JSON, $-0.5$ if omitted.
3. **Blast Radius Penalty ($R_{\text{penalty}}$):** $-1.5$ per policy violation (e.g. attempting `kubectl delete namespace` or touching `kube-system`).
4. **Step Efficiency ($R_{\text{efficiency}}$):** $-0.03 \times \text{steps}$ (capped at $-0.25$) to encourage concise diagnostic paths.

$$R_{\text{total}} = R_{\text{health}} + R_{\text{diagnosis}} - R_{\text{penalty}} - R_{\text{efficiency}}$$

---

## Full Cloud Deployment on AWS

The project includes a complete, cloud-native architecture to generate data, train, evaluate, and operate the Large Reasoning Model autonomously on AWS.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1. SYNTHETIC DATA ENGINE                               │
│  Amazon Bedrock (Claude 3.5 / Llama 3) ──► S3 Data Lake (s3://lrm-bucket/sft-data/)     │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 2. RLVR TRAINING CLUSTER                               │
│  Amazon SageMaker Training (ml.g5.12xlarge / ml.p4d)                                  │
│    ├── Policy Model (Trainable Qwen-2.5-Coder-7B)                                      │
│    ├── Frozen Reference Model                                                          │
│    └── GRPO Optimizer                                                                  │
│            ▲                                      │                                    │
│            │ (Rewards & Observations)             │ (Tool calls: kubectl)              │
│            └──────────────────┐                   ▼                                    │
│                               │      Amazon EKS (VPC-peered Sandbox Cluster)           │
│                               │        ├── Namespace: worker-1 (OOM Injection)         │
│                               │        ├── Namespace: worker-2 (Selector Bug)          │
│                               └─────── └── Verifier Probes (HTTP 200, Pod Status)      │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼ (Model Weights)
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                            3. PRODUCTION AUTONOMOUS SRE LOOP                           │
│  Amazon CloudWatch Alarm ──► EventBridge ──► Lambda Orchestrator                       │
│                                                      │                                 │
│                                                      ▼                                 │
│                                          SageMaker Real-Time Endpoint                  │
│                                          (vLLM Container serving tuned LRM)            │
│                                                      │                                 │
│                                                      ▼                                 │
│                                          EKS Production Cluster                        │
│                                          (Self-Healing & Remediation)                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### AWS Components & Directory Structure

```
large-reasoning-model-rlvr-sre/
├── aws/
│   ├── terraform/                     # Complete Infrastructure as Code
│   │   ├── main.tf                    # VPC, Subnets, S3 Buckets, Security Groups
│   │   ├── eks.tf                     # Amazon EKS cluster and Spot node group for RLVR
│   │   ├── sagemaker.tf               # SageMaker IAM role and VPC endpoints
│   │   ├── eventbridge_lambda.tf      # CloudWatch Alarm, EventBridge rule & Lambda
│   │   ├── variables.tf               # Configurable region, cluster name, instance types
│   │   └── outputs.tf                 # Subnet IDs, Role ARNs, Bucket names
│   ├── bedrock/
│   │   └── generate_synthetic_data.py # Distills IRAC traces via Claude 3.5 Sonnet to S3
│   ├── sagemaker/
│   │   ├── train_eks_rlvr.py          # SageMaker entrypoint running RLVR across EKS
│   │   ├── launch_training.py         # Submits multi-GPU training job inside VPC
│   │   └── deploy_endpoint.py         # Deploys fine-tuned model via vLLM DLC
│   └── lambda_sre/
│       └── orchestrator.py            # Serverless SRE agent triggered by CloudWatch
├── requirements.txt                   # Installed into SageMaker container
├── train_sagemaker.py                 # Standalone SageMaker entrypoint
└── launch_sagemaker.py                # Standalone SageMaker launcher
```

---

### Step-by-Step AWS Deployment Guide

#### 1. Provision Infrastructure via Terraform
Provisions the VPC, S3 data/model buckets, private subnets, EKS sandbox cluster, and IAM roles:

```powershell
cd aws/terraform
terraform init
terraform apply -auto-approve
```

Note the outputs (`sagemaker_execution_role_arn`, `private_subnets`, `data_lake_bucket`).

#### 2. Distill Synthetic Cold-Start Data with Amazon Bedrock
Uses **Claude 3.5 Sonnet** via Bedrock Converse API to generate verified reasoning traces and streams them to S3:

```powershell
uv run python aws/bedrock/generate_synthetic_data.py `
    --model-id "anthropic.claude-3-5-sonnet-20241022-v2:0" `
    --s3-bucket "<DATA_LAKE_BUCKET>" `
    --region "us-east-1"
```

#### 3. Run Distributed RLVR Training on SageMaker + EKS
Submits the training job into the private VPC subnets with access to the Amazon EKS cluster:

```powershell
uv run python aws/sagemaker/launch_training.py `
    --role-arn "<SAGEMAKER_ROLE_ARN>" `
    --subnets "<SUBNET_1>,<SUBNET_2>" `
    --security-groups "<SAGEMAKER_SG_ID>" `
    --eks-cluster "lrm-sre-sandbox-eks" `
    --instance-type "ml.g5.12xlarge" `
    --spot
```

* Each rollout worker operates in an isolated ephemeral namespace (`sandbox-worker-01`, etc.) on EKS.
* Checkpoints are automatically synced to `s3://<MODELS_BUCKET>/output/model.tar.gz`.

#### 4. Deploy to SageMaker Real-Time Endpoint (vLLM)
Deploys the trained reasoning model using the AWS vLLM Deep Learning Container on an `ml.g5.2xlarge` instance:

```powershell
uv run python aws/sagemaker/deploy_endpoint.py `
    --model-s3-uri "s3://<MODELS_BUCKET>/output/model.tar.gz" `
    --role-arn "<SAGEMAKER_ROLE_ARN>" `
    --endpoint-name "lrm-sre-reasoning-endpoint" `
    --instance-type "ml.g5.2xlarge"
```

#### 5. Autonomous SRE Closed-Loop Healing (CloudWatch & Lambda)
* When a pod enters `CrashLoopBackOff` or `OOMKilled`, CloudWatch triggers an alarm.
* EventBridge dispatches the event to [`aws/lambda_sre/orchestrator.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/lambda_sre/orchestrator.py).
* Lambda queries the SageMaker endpoint, extracts the `<action>` JSON patch, applies it to EKS, and verifies recovery.

Test the Lambda orchestrator locally anytime:
```powershell
uv run python aws/lambda_sre/orchestrator.py
```

---

## Alternative Cloud Architectures

In addition to the default Qwen on SageMaker setup, the repository provides three dedicated alternative architectures that keep the base version completely intact:

### Architecture Comparison Matrix

| Dimension | Base Architecture (`aws/sagemaker`) | Option 1: Pure Bedrock Serverless | Option 2: SageMaker $\to$ Bedrock CMI | Option 3: Bedrock Serverless + RAG |
| :--- | :--- | :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-Coder-7B` | `Claude 3.5 Haiku` / `Llama 3.3 70B` | Custom `Meta-Llama-3.1-8B` | `Claude 3.5 Haiku` / `Llama 3.3 70B` |
| **Training Pipeline** | SageMaker Distributed GRPO | Zero Training (In-Context Prompting) | SageMaker GRPO $\to$ Safetensors Export | Zero Training (In-Context Prompting) |
| **Inference Hosting** | Dedicated SageMaker vLLM Endpoint (`ml.g5.2xlarge`) | Serverless Bedrock Converse API | Bedrock Custom Model Import (CMI) | Serverless Bedrock Converse API |
| **Idle Infrastructure Cost** | ~\$1.21/hour (~\$870/month 24/7) | **\$0.00/hour** (pay strictly per token) | **\$0.00/hour** (pay strictly per token) | **\$0.00/hour** (pay strictly per token) |
| **Proprietary Knowledge** | None (Generic synthetic SRE) | Public foundation model knowledge only | Injected directly into weights via RLVR | **Dynamically retrieved from SRE Runbooks (RAG)** |
| **Maintenance Effort** | High (GPU endpoints, vLLM DLC management) | Low (Pure serverless event-driven Lambda) | Medium (Periodic model training & re-import) | **Low** (Just edit Markdown runbooks in S3/Git) |
| **Safety Guarantees** | Blast-radius penalty in RL loss ($R_{\text{penalty}}$) | System prompt guards & IAM role boundaries | Blast-radius penalty in RL loss ($R_{\text{penalty}}$) | **Deterministic Runbook SOP Grounding + IAM** |

---

### Option 1: Pure Bedrock Serverless SRE Agent (Zero GPU Endpoints)

Eliminates SageMaker GPU endpoints entirely. The agent runs serverlessly on Amazon Bedrock using **Claude 3.5 Haiku** (`anthropic.claude-3-5-haiku-20241022-v1:0`) or **Meta Llama 3.3 70B** (`meta.llama3-3-70b-instruct-v1:0`).

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               OPTION 1: PURE BEDROCK SERVERLESS AUTONOMOUS SRE ARCHITECTURE             │
│                       (100% Serverless - Zero GPU Endpoints to Manage)                 │
└────────────────────────────────────────────────────────────────────────────────────────┘

     1. INCIDENT OCCURS
  ┌───────────────────────┐
  │      Amazon EKS       │
  │  (Production Cluster) │
  │   CrashLoopBackOff    │
  └───────────┬───────────┘
              │ 2. Metrics / Log Stream
              ▼
  ┌───────────────────────┐
  │   Amazon CloudWatch   │
  │  (ContainerInsights)  │
  │     Alarm: ALARM      │
  └───────────┬───────────┘
              │ 3. State Change Event
              ▼
  ┌───────────────────────┐
  │   Amazon EventBridge  │
  │   (Event Router Rule) │
  └───────────┬───────────┘
              │ 4. Dispatches Alert Payload
              ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │                          AWS Lambda (SRE Orchestrator)                               │
  │                  (aws/option1_bedrock_serverless/orchestrator_bedrock.py)             │
  │                                                                                      │
  │   a. Harvests diagnostic events & pod logs from Amazon EKS API                       │
  │   b. Formulates diagnostic prompt with <context>                                     │
  │   c. Invokes Bedrock Converse API                                                    │
  │                                                                                      │
  │         │                                                              ▲             │
  │         │ 5. Prompt (<context>...)         6. CoT & Tool Response      │             │
  │         ▼                                                              │             │
  │  ┌────────────────────────────────────────────────────────────────┐    │             │
  │  │                     Amazon Bedrock                             │    │             │
  │  │      Claude 3.5 Haiku / Sonnet OR Meta Llama 3.3 70B           │────┘             │
  │  │                                                                │                  │
  │  │  Generates:                                                    │                  │
  │  │   <think>Diagnostic hypothesis & self-correction</think>       │                  │
  │  │   <action>{"tool": "kubectl", "cmd": "patch deploy..."}</action>│                 │
  │  │   <verdict>{"root_cause": "OOM", "status": "RESOLVED"}</verdict>│                 │
  │  └────────────────────────────────────────────────────────────────┘                  │
  │                                                                                      │
  │   d. Safety Gate: Validates blast-radius (blocks "delete ns", etc.)                  │
  │   e. Applies kubectl patch to EKS & runs health probe check                          │
  └───────────────────────────────────────────┬──────────────────────────────────────────┘
                                              │
                                              │ 7. Remediation & Verification Probe
                                              ▼
                                 ┌─────────────────────────┐
                                 │       Amazon EKS        │
                                 │   Pod Restarted (256Mi) │
                                 │   Probe returns 200 OK  │
                                 └─────────────────────────┘
```

* Directory: [`aws/option1_bedrock_serverless/`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option1_bedrock_serverless)
* **Components:**
  * [`orchestrator_bedrock.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option1_bedrock_serverless/orchestrator_bedrock.py): Event-driven Lambda orchestrator that queries Bedrock directly for reasoning and applies remediation.
  * [`serverless_bedrock.tf`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option1_bedrock_serverless/serverless_bedrock.tf): Independent Terraform definition provisioning Lambda + EventBridge + Bedrock IAM (zero EC2/SageMaker costs).

#### Running / Testing Option 1:
```powershell
uv run python aws/option1_bedrock_serverless/orchestrator_bedrock.py
```

---

### Option 2: Meta-Llama on SageMaker $\rightarrow$ Amazon Bedrock Custom Model Import (CMI)

Trains an open-weights model (**`meta-llama/Llama-3.1-8B`**) using SageMaker GRPO, and imports the fine-tuned weights directly into Amazon Bedrock via **Custom Model Import (CMI)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│       OPTION 2: META-LLAMA ON SAGEMAKER -> BEDROCK CUSTOM MODEL IMPORT (CMI)           │
│         (Custom Multi-GPU RLVR Training with Serverless Bedrock Serving)               │
└────────────────────────────────────────────────────────────────────────────────────────┘

 ════════════════════════════════════════════════════════════════════════════════════════
  PHASE 1: MULTI-GPU RLVR / GRPO TRAINING ON SAGEMAKER (aws/option2_sagemaker_to_bedrock_cmi/)
 ════════════════════════════════════════════════════════════════════════════════════════

   Base Weights (Hugging Face)
    [ meta-llama/Llama-3.1-8B ]
               │
               ▼
   ┌──────────────────────────────────────────────────────────────────────────────────┐
   │             Amazon SageMaker Training Job (ml.g5.12xlarge / Spot)                │
   │                                                                                  │
   │   Active Policy Model (Llama-3.1-8B) ◄───► Frozen Reference Model (Llama-3.1-8B) │
   │                             │                         │                          │
   │                Rollout 1    │        ...              │ Rollout G                │
   │                    ▼        │                         ▼                          │
   │             [ Worker 1 ]    │                  [ Worker G ]                      │
   │                   │         │                         │                          │
   │                   ▼         ▼                         ▼                          │
   │         ┌─────────────────────────────────────────────────────────┐              │
   │         │               Amazon EKS (Sandbox Cluster)              │              │
   │         │    ns/sandbox-worker-01      ...   ns/sandbox-worker-G  │              │
   │         │   (Fault Injection: OOM)          (Selector Mismatch)   │              │
   │         │               │                             │           │              │
   │         │               ▼                             ▼           │              │
   │         │          HTTP Probe                    HTTP Probe       │              │
   │         │           Exit Code                     Exit Code       │              │
   │         └─────────────────────────────────────────────────────────┘              │
   │                                     │                                            │
   │   Rewards R_1 ... R_G ──────────────┘                                            │
   │                                                                                  │
   │   Advantages A_i = (R_i - mean(R)) / std(R)                                      │
   │   GRPO Loss = Policy Gradient + Beta * KL_Divergence                             │
   │   optimizer.step() (Model Weights Updated)                                       │
   │                                                                                  │
   │   Save weights in Safetensors format ────────────────────────────────┐           │
   └──────────────────────────────────────────────────────────────────────┼───────────┘
                                                                          │
                                                                          ▼
                                                            ┌────────────────────────┐
                                                            │       Amazon S3        │
                                                            │ s3://lrm-models/llama/ │
                                                            │  (Safetensors weights) │
                                                            └─────────────┬──────────┘
 ═════════════════════════════════════════════════════════════════════════╪══════════════
  PHASE 2: BEDROCK CUSTOM MODEL IMPORT (import_to_bedrock_cmi.py)         │
 ═════════════════════════════════════════════════════════════════════════╪══════════════
                                                                          ▼
                                                       ┌─────────────────────────────┐
                                                       │    Bedrock Import Job       │
                                                       │ create_model_import_job()   │
                                                       └──────────────┬──────────────┘
                                                                      │ Validates & Registers
                                                                      ▼
                                                       ┌─────────────────────────────┐
                                                       │   Amazon Bedrock Registry   │
                                                       │   Private Custom Model:     │
                                                       │   arn:aws:bedrock:...:      │
                                                       │   imported-model/lrm-llama  │
                                                       └──────────────┬──────────────┘
 ═════════════════════════════════════════════════════════════════════╪════════════════
  PHASE 3: PRODUCTION SERVERLESS INFERENCE (invoke_custom_bedrock_model.py)
 ═════════════════════════════════════════════════════════════════════╪════════════════
                                                                      │
  EKS Alarm ──► EventBridge ──► Lambda Orchestrator                   │
                                      │                               │
                                      │ Invoke custom model via ARN   │
                                      └───────────────────────────────┘
```

* Directory: [`aws/option2_sagemaker_to_bedrock_cmi/`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option2_sagemaker_to_bedrock_cmi)
* **Components:**
  * [`train_llama_sagemaker.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option2_sagemaker_to_bedrock_cmi/train_llama_sagemaker.py): SageMaker entrypoint trained on EKS rollouts, outputting Safetensors weights.
  * [`launch_llama_training.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option2_sagemaker_to_bedrock_cmi/launch_llama_training.py): Submits the Llama training job to SageMaker GPU instances.
  * [`import_to_bedrock_cmi.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option2_sagemaker_to_bedrock_cmi/import_to_bedrock_cmi.py): Automates `bedrock.create_model_import_job()` to register the S3 weights as a private Bedrock model.
  * [`invoke_custom_bedrock_model.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option2_sagemaker_to_bedrock_cmi/invoke_custom_bedrock_model.py): Queries the imported model using standard Bedrock Converse API via its custom model ARN.

#### Running Option 2 Pipeline:
```powershell
# 1. Train Llama on SageMaker
uv run python aws/option2_sagemaker_to_bedrock_cmi/launch_llama_training.py `
    --role-arn "<SAGEMAKER_ROLE_ARN>" `
    --model-id "meta-llama/Llama-3.1-8B" `
    --spot

# 2. Import trained weights into Amazon Bedrock
uv run python aws/option2_sagemaker_to_bedrock_cmi/import_to_bedrock_cmi.py `
    --model-s3-uri "s3://<MODELS_BUCKET>/output/model/" `
    --role-arn "<BEDROCK_IMPORT_ROLE_ARN>"

# 3. Test inference via Bedrock Converse API
uv run python aws/option2_sagemaker_to_bedrock_cmi/invoke_custom_bedrock_model.py `
    --model-arn "arn:aws:bedrock:us-east-1:123456789012:imported-model/<MODEL_NAME>"
```

---

### Option 3: Bedrock Serverless SRE Agent Grounded with Knowledge Bases (RAG)

Combines the **\$0 idle cost** of Option 1 with **deep organizational intelligence** without fine-tuning. It retrieves your team's internal Markdown runbooks, postmortems, and service catalogs from an **Amazon Bedrock Knowledge Base** and injects them alongside live cluster telemetry into the prompt.

```
+-----------------------------------------------------------------------------------------+
|                                  KNOWLEDGE INGESTION (Offline)                          |
|                                                                                         |
|   +--------------------+       Sync        +----------------------------------------+   |
|   | S3 Bucket:         | ----------------> | Amazon Bedrock Knowledge Base          |   |
|   | - Runbooks (.md)   |                   | - Embeddings (Titan Text v2)           |   |
|   | - Postmortems      |                   | - Vector Store (OpenSearch Serverless) |   |
|   | - Service Catalogs |                   +-------------------+--------------------+   |
+---+--------------------+---------------------------------------|------------------------+
                                                                 |
=================================================================|=========================
                                     INCIDENT RUNTIME            |
=================================================================|=========================
                                                                 |
   +----------------------+                                      |
   | Amazon EKS / CW      |                                      |
   | (CrashLoopBackOff)   |                                      |
   +----------+-----------+                                      |
              | (1) Alarm Trigger                                |
              v                                                  |
   +----------------------+                                      |
   |  Amazon EventBridge  |                                      |
   +----------+-----------+                                      |
              | (2) Event Payload                                |
              v                                                  |
+----------------------------------------------------+           |
|            AWS Lambda Orchestrator                 |           |
|  (aws/option3_bedrock_rag/                         |           |
|   orchestrator_bedrock_rag.py)                     |           |
|                                                    |           |
|   1. Parse Alert Signature                         |           |
|   2. Query Knowledge Base for Matching Runbook     |<----------+ (3) Retrieve Top
|   3. Fetch Real-time Pod Logs via kubectl          |                 Runbook Context
|   4. Assemble Enriched SRE Prompt:                 |
|      [Alert + Runbook Excerpt + Live Logs]         |
+---------------------+------------------------------+
                      |
                      | (4) Invoke with Runbook Grounding
                      v
+----------------------------------------------------+
|               Amazon Bedrock Model                 |
|             (Claude 3.5 Haiku / Sonnet)            |
|                                                    |
|   - Follows company-approved runbook steps         |
|   - Zero hallucination of internal endpoints       |
|   - Billed strictly per token ($0 idle cost)       |
+---------------------+------------------------------+
                      |
                      | (5) Targeted Fix Action
                      v
+----------------------------------------------------+
|              Amazon EKS Cluster                    |
|      (Safely healed per internal SOP)              |
+----------------------------------------------------+
```

* Directory: [`aws/option3_bedrock_rag/`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option3_bedrock_rag)
* **Components:**
  * [`runbooks/`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option3_bedrock_rag/runbooks): Production Markdown runbooks (`runbook_oom_killed.md`, `runbook_502_bad_gateway.md`, `runbook_crashloop_config.md`).
  * [`kb_retriever.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option3_bedrock_rag/kb_retriever.py): Semantic retriever wrapping `bedrock-agent-runtime.retrieve()` with automatic local markdown fallback.
  * [`orchestrator_bedrock_rag.py`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option3_bedrock_rag/orchestrator_bedrock_rag.py): Lambda handler fusing dynamic cluster state with static runbook SOPs to invoke Bedrock.
  * [`bedrock_rag.tf`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option3_bedrock_rag/bedrock_rag.tf): Terraform IaC deploying S3 runbooks bucket, Bedrock Knowledge Base, OpenSearch Serverless collection, IAM roles, and Lambda.

#### Deep Dive: How RAG Bridges the Gap with Fine-Tuning
While Option 1 is 100% serverless, it only knows general Kubernetes concepts. By introducing **Amazon Bedrock Knowledge Bases**, Option 3 equips the agent with:
* **Company SOPs & Playbooks:** Enforces internal guidelines (e.g., maximum tier memory allocations, which services permit automated rollouts).
* **Incident Postmortems:** Matches recurring alert signatures against historical root causes.
* **Prohibited Operations:** Grounding the prompt in runbook safety rules ensures the model avoids destructive operations (like deleting namespaces or persistent storage).
* **Zero Model Retraining:** When an internal architecture or runbook changes, engineers simply commit an updated Markdown document to S3/Git. The Bedrock Knowledge Base syncs automatically without expensive GPU retraining cycles.

#### Local Verification (Zero AWS Credentials Required):
The retriever includes an automatic local fallback that searches [`aws/option3_bedrock_rag/runbooks/`](file:///c:/Users/alan_/Projetos/labs/ai-experiments/large-reasoning-model-rlvr-sre/aws/option3_bedrock_rag/runbooks) using BM25/keyword overlap if AWS credentials or an active Knowledge Base ID are absent.

```powershell
uv run python aws/option3_bedrock_rag/orchestrator_bedrock_rag.py
```

Expected trace:
```text
[*] Harvested Live Cluster Telemetry:
ALERT: EKS-Pod-Degraded-Alarm transitioned to ALARM.
Cluster: lrm-sre-sandbox-eks | Namespace: sandbox-app | Pod: payment-processor-68df9-x2p8
Status: CrashLoopBackOff | Last Termination Reason: OOMKilled (Exit Code: 137)

[*] Querying Bedrock Knowledge Base for: 'EKS-Pod-Degraded-Alarm OOMKilled Exit Code 137 CrashLoopBackOff memory limit'...
  -> Found relevant runbook [1] (Source: .../runbooks/runbook_oom_killed.md, Score: 41.00)
  -> Found relevant runbook [2] (Source: .../runbooks/runbook_crashloop_config.md, Score: 12.00)

[*] Model Reasoning & Action Output:
<think>
Alert shows pod 'payment-processor-68df9-x2p8' terminated with OOMKilled (Exit Code 137).
According to the official runbook (runbook_oom_killed.md):
- Root cause category is MEMORY_LIMIT_TOO_LOW.
- Standard remediation procedure states: update cgroup memory limit to at least 256Mi for microservices.
- Safety prohibition: NEVER delete namespace or PVCs.
Remediation plan: Patch deployment payment-processor memory limits to 256Mi and requests to 128Mi.
</think>
<action>
{"tool": "kubectl", "cmd": "patch deployment payment-processor -n sandbox-app --patch '{\"spec\":{\"template\":{\"spec\":{\"containers\":[{\"name\":\"worker\",\"resources\":{\"limits\":{\"memory\":\"256Mi\"},\"requests\":{\"memory\":\"128Mi\"}}}]}}}}'"}
</action>
<verdict>
{"root_cause": "MEMORY_LIMIT_TOO_LOW", "status": "RESOLVED", "runbook_cited": "runbook_oom_killed.md"}
</verdict>
```

#### Deploying Option 3 to AWS with Terraform:
```powershell
cd aws/option3_bedrock_rag

# Initialize and deploy S3 runbook bucket, Bedrock Knowledge Base, OpenSearch Serverless, and Lambda
terraform init
terraform plan -out=tfplan
terraform apply tfplan

# Once deployed, note the outputs:
# - bedrock_kb_id: The ID of the Bedrock Knowledge Base
# - lambda_rag_arn: The ARN of the SRE orchestrator Lambda function
cd ../..
```

#### Testing Live with Real Bedrock Knowledge Base:
```powershell
# Set environment variables with your deployed resources
$env:BEDROCK_KB_ID = "<YOUR_BEDROCK_KNOWLEDGE_BASE_ID>"
$env:BEDROCK_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"
$env:AWS_REGION = "us-east-1"

# Run the orchestrator
uv run python aws/option3_bedrock_rag/orchestrator_bedrock_rag.py
```


