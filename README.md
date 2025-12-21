# Adaptive Test-Time Scaling (ATTS) - Research Implementation

This repository contains the experimental validation code for the research proposal:

**"Adaptive Self-Verifiable Reasoning: A Proposed Architecture for Efficient LLM Reasoning with Dynamic Test-Time Compute Allocation"**

## Overview

The paper proposes three interconnected architectural innovations to address the "token tax" problem in LLM reasoning:

1. **Unified Self-Verification Architecture (USVA)** - Domain-agnostic verification framework
2. **Dynamic Sparse Attention (DSA-2)** - Query-dependent token budget allocation
3. **Adaptive Test-Time Scaling (ATTS)** - Difficulty-based compute routing

This implementation focuses on validating the **ATTS** mechanism (Section 2.3) and incorporates **USVA-inspired self-verification** (Section 2.1) using existing models, without requiring model training.

## Key Hypothesis

**Can we achieve 40-60% token reduction while maintaining accuracy within 2-5% of full-compute baselines by adaptively routing problems to appropriate compute tiers?**

## Architecture

### ATTS Workflow

```
Input Problem
     ↓
[Difficulty Estimation] → difficulty score d ∈ [1, 10]
     ↓
[Mode Selection]
  ├─ d < 4  → Direct Mode (~150 tokens)
  ├─ 4 ≤ d < 7 → Thinking Mode (~500 tokens)
  └─ d ≥ 7  → Deep Mode (~1000 tokens)
     ↓
[Solution Generation]
     ↓
[Self-Verification] → verification score v
     ↓
[Escalation Check] → if v < τ, escalate to next tier
     ↓
Output: Solution + Metrics
```

### Three Reasoning Modes

| Mode | Trigger | Behavior | Avg Tokens |
|------|---------|----------|------------|
| **Direct** | d < 4 | Concise, direct answer | ~150 |
| **Thinking** | 4 ≤ d < 7 | Step-by-step reasoning | ~500 |
| **Deep** | d ≥ 7 | Full verification loop | ~1000 |

## Experiment Design

Based on **Section 4 (Proposed Experimental Validation)** of the paper:

### Protocol

1. **Dataset**: 25 math problems (easy/medium/hard stratified)
2. **Difficulty Classification**: Model rates 1-10 WITHOUT solving
3. **Adaptive Routing**: Route to Direct/Thinking/Deep based on threshold
4. **Baseline**: Always-Deep mode for comparison
5. **Self-Verification**: USVA-inspired scoring using 4 rubrics:
   - Logical Coherence (LC)
   - Factual Correctness (FC)
   - Completeness (CM)
   - Goal Alignment (GA)
6. **Escalation**: If verification score < threshold, escalate to next tier

### Metrics

- **Accuracy**: % correct answers
- **Token Efficiency**: Average tokens per problem
- **Token Savings**: % reduction vs baseline
- **Difficulty Classification Accuracy**: How well difficulty prediction matches actual difficulty
- **Escalation Rate**: % of problems that triggered escalation
- **Mode Distribution**: Breakdown by easy/medium/hard

## Repository Structure

```
ATTS_Complete_Free_Package/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── atts_experiment_local.py      # Main experiment script (enhanced)
├── paper/                        # Research paper
│   └── Adaptive_Self_Verifiable_Reasoning_Proposal.pdf
├── data/                         # Experimental datasets
│   └── math_problems.json        # 25 math problems (stratified)
├── results/                      # Experimental results
│   └── example_results.json      # Sample results
└── docs/                         # Additional documentation
```

## Setup

### Prerequisites

1. **Docker** (for Ollama)
2. **Python 3.8+**

### Installation

```bash
# 1. Install Ollama via Docker
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 2. Pull a model (Qwen 2.5 3B recommended for laptops)
docker exec -it ollama ollama pull qwen2.5:3b-instruct

# 3. Install Python dependencies
pip install -r requirements.txt
```

### Verify Ollama is Running

```bash
# Test the model
docker exec -it ollama ollama run qwen2.5:3b-instruct
# Type "hello" to test, then Ctrl+D to exit
```

## Usage

### Run Full Experiment

```bash
python atts_experiment_local.py --model qwen2.5:3b-instruct
```

### Command-line Options

```bash
python atts_experiment_local.py --help

Options:
  --model MODEL       Ollama model name (default: qwen2.5:3b-instruct)
  --dataset DATASET   Path to dataset (default: data/math_problems.json)
  --escalation        Enable uncertainty-triggered escalation (default: True)
  --threshold FLOAT   Verification score threshold for escalation (default: 0.6)
```

### Expected Output

```
🏠 ATTS Local Experiment (Ollama: qwen2.5:3b-instruct)
==================================================
✅ Ollama connected

Loaded 25 problems

ATTS: 100%|████████████████| 25/25 [05:30<00:00, 13.2s/it]
Baseline: 100%|██████████████| 25/25 [08:45<00:00, 21.0s/it]

==================================================
📊 RESULTS
==================================================

Baseline: 96.0% accuracy, 459 avg tokens
ATTS:     92.0% accuracy, 251 avg tokens

💰 Token Savings: 45.3%
📈 Mode Distribution: {'thinking': 14, 'direct': 9, 'deep': 2}
📊 Escalation Rate: 12.0%

==================================================
✅ HYPOTHESIS SUPPORTED!
==================================================
```

## Validation Results

### Validated Paper Sections

- ✅ **Section 2.3 (ATTS)**: Adaptive routing based on difficulty
- ✅ **Section 2.3.1**: Difficulty estimation
- ✅ **Section 2.3.2**: Compute allocation policy
- ✅ **Section 2.3.3**: Uncertainty-triggered escalation
- ✅ **Section 2.1** (partial): USVA-inspired self-verification
- ✅ **Section 3.1**: Token efficiency estimates
- ✅ **Section 4.1**: Simulation protocol

### Not Validated (Requires Training)

- ⏸️ **Section 2.1** (full USVA): Requires custom training
- ⏸️ **Section 2.2** (DSA-2): Requires architecture changes
- ⏸️ **Section 2.4**: Distilled verification knowledge

## Results

Example results from local testing (qwen2.5:3b-instruct):

| Metric | Baseline (Always-Deep) | ATTS | Improvement |
|--------|------------------------|------|-------------|
| Accuracy | 96.0% | 92.0% | -4.0% |
| Avg Tokens | 459 | 251 | **45.3% reduction** |
| Easy Problems | 100% (7/7) | 100% (7/7) | 0% |
| Medium Problems | 100% (10/10) | 90% (9/10) | -10% |
| Hard Problems | 87.5% (7/8) | 87.5% (7/8) | 0% |

**Conclusion**: Hypothesis supported - achieved >40% token savings with <5% accuracy drop.

## Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores (for qwen2.5:3b)
- **Recommended**: 16GB RAM, 8 CPU cores (for larger models)
- **GPU**: Not required (CPU-only works fine for 3B models)

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if container is running
docker ps

# Start container if stopped
docker start ollama

# Check logs
docker logs ollama
```

### Model Not Found

```bash
# List available models
docker exec -it ollama ollama list

# Pull the model
docker exec -it ollama ollama pull qwen2.5:3b-instruct
```

### Out of Memory

Try a smaller model:
```bash
docker exec -it ollama ollama pull qwen2.5:1.5b-instruct
python atts_experiment_local.py --model qwen2.5:1.5b-instruct
```

## Citation

If you use this code, please cite the paper:

```bibtex
@article{adaptive_self_verifiable_2025,
  title={Adaptive Self-Verifiable Reasoning: A Proposed Architecture for Efficient LLM Reasoning with Dynamic Test-Time Compute Allocation},
  author={[Your Name]},
  journal={Research Proposal},
  year={2025}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

This is a research proposal implementation. Contributions welcome:

1. Additional validation experiments
2. Support for more model backends
3. Enhanced verification rubrics
4. Larger/more diverse datasets

## Contact

For questions or collaboration: [your.email@institution.edu]

## Acknowledgments

- Ollama for local model serving
- Qwen team for efficient small models
- Research inspired by recent work on adaptive compute and self-verification
