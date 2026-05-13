# PromptOps Benchmarks

Real AWS Bedrock benchmarks demonstrating PromptOps performance and accuracy.

## Prerequisites

1. **Python 3.9+** with PromptOps installed:
   ```bash
   pip install substrai-promptops[aws]
   ```

2. **AWS credentials** configured with Bedrock access:
   ```bash
   aws configure
   # Or use SSO:
   aws sso login
   ```

3. **Bedrock model access** — ensure `anthropic.claude-3-haiku-20240307-v1:0` is enabled in your AWS account (us-east-1).

## Running the Benchmark

```bash
# From the repo root
cd benchmarks
python run_aws_benchmark.py
```

## What It Tests

| Benchmark | Description |
|-----------|-------------|
| **1. Prompt Resolution** | Measures template rendering overhead (100 iterations) |
| **2. Bedrock Invocation** | End-to-end latency with real Claude 3 Haiku calls |
| **3. Cost Estimation** | Compares estimated vs actual Bedrock costs |
| **4. Model Router** | Measures routing decision speed (1000 iterations) |
| **5. Quality Evaluation** | Runs evaluators on real LLM output |
| **6. Adversarial Detection** | Tests injection detection on real responses |
| **7. Fallback Chain** | Tests automatic model failover |

## Expected Output

```
======================================================================
PROMPTOPS AWS BENCHMARK
======================================================================

--- Benchmark 1: Prompt Resolution & Rendering Overhead ---
  Template rendering (100 iterations):
    Mean:   0.002 ms
    ...

BENCHMARK SUMMARY
  PromptOps overhead per invocation:
    Template rendering:  0.002 ms
    Model routing:       0.0043 ms
    Total framework:     0.006 ms
  Framework overhead: 0.00% of total latency
```

## Results

See [RESULTS.md](RESULTS.md) for the full benchmark results from our latest run.

## Cost

Running the full benchmark costs approximately **$0.002** (less than 1 cent) — it makes ~7 Bedrock Haiku calls with minimal tokens.

## Key Findings

- **Framework overhead: 0.006ms** — essentially zero impact on LLM call latency
- **Model routing: 4.3 microseconds** — negligible decision time
- **Injection detection: 100%** — correctly blocks adversarial inputs
- **Fallback chain: works** — auto-recovers from model failures
