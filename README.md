# ATTS: Adaptive Test-Time Scaling

A framework for optimizing inference compute allocation in language models through difficulty-adaptive scaling.

## Key Results

| Metric | ATTS | Baseline | Improvement |
|--------|------|----------|-------------|
| Token Usage | ~650 avg | ~850 avg | **20-25% savings** |
| Accuracy | 94-96% | 95-97% | Within 2% |
| Pareto Efficient | Yes | N/A | Optimal trade-off |

## Quick Start

### Prerequisites

```bash
# 1. Start Ollama (Docker)
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# 2. Pull a model
docker exec -it ollama ollama pull qwen2.5:3b-instruct

# 3. Install dependencies
pip install -r requirements.txt
```

### Run Experiments

```bash
# Quick test (5 problems, ~2 min)
python run_atts.py --quick-test

# Standard experiment (25 problems, ~15 min)
python run_atts.py --max-problems 25

# Full experiment with refinement (100 problems, ~1 hour)
python run_atts.py --max-problems 100 --enable-refinement

# Verbose output with decision traces
python run_atts.py --max-problems 10 --verbose

# See all options
python run_atts.py --help
```

### Example Output

```
============================================================
                   COMPREHENSIVE RESULTS
============================================================
Baseline: 96.0% accuracy, 847 avg tokens
ATTS:     94.0% accuracy, 672 avg tokens

Token Savings: 20.7%
Mode Distribution: {'direct': 8, 'thinking': 12, 'deep': 5}
Escalation Rate: 32.0%

PARETO FRONTIER ANALYSIS
Pareto Improvement: YES
```

## Methodology

ATTS implements a 6-stage adaptive pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                      ATTS WORKFLOW                          │
├─────────────────────────────────────────────────────────────┤
│  1. DIFFICULTY ESTIMATION                                   │
│     └─ Pass@k sampling → d ∈ [1,10], uncertainty σ          │
│                                                             │
│  2. MODE SELECTION                                          │
│     └─ d < 4: DIRECT | d < 7: THINKING | else: DEEP        │
│                                                             │
│  3. SOLUTION GENERATION                                     │
│     └─ Mode-specific prompt, token budget                   │
│                                                             │
│  4. USVA VERIFICATION                                       │
│     └─ Rubrics: LC, FC, CM, GA → v ∈ [0,1]                 │
│                                                             │
│  5. ESCALATION (if v < 0.80)                               │
│     └─ Upgrade mode, re-solve, re-verify                    │
│                                                             │
│  6. REFINEMENT (deep mode only)                            │
│     └─ Critic → Meta-verify → Refine loop                   │
└─────────────────────────────────────────────────────────────┘
```

### Compute Modes

| Mode | Token Budget | Use Case |
|------|--------------|----------|
| DIRECT | 150 | Single-step arithmetic |
| THINKING | 500 | Multi-step reasoning |
| DEEP | 1000 | Competition-level problems |

### USVA Verification Rubrics

- **LC** (Logical Coherence): Do reasoning steps follow logically?
- **FC** (Factual Correctness): Are calculations accurate?
- **CM** (Completeness): Are all aspects addressed?
- **GA** (Goal Alignment): Is there a clear final answer?

## Project Structure

```
ATTS/
├── run_atts.py              # Main entry point
├── requirements.txt         # Dependencies
├── pyproject.toml          # Package configuration
│
├── config/
│   ├── prompts.yaml         # Prompt templates
│   ├── thresholds.yaml      # Algorithm thresholds
│   └── settings.yaml        # General settings
│
├── src/atts/
│   ├── domain/              # Core entities (no external dependencies)
│   │   ├── entities.py      # Problem, Solution, WorkflowResult
│   │   ├── value_objects.py # ComputeMode, RubricScores, DecisionTrace
│   │   └── exceptions.py    # Custom exceptions
│   │
│   ├── ports/               # Abstract interfaces
│   │   ├── model_caller.py  # IModelCaller interface
│   │   ├── config_loader.py # IConfigLoader interface
│   │   └── repository.py    # IResultRepository interface
│   │
│   ├── adapters/            # Concrete implementations
│   │   ├── ollama_adapter.py      # Ollama LLM integration
│   │   ├── yaml_config_loader.py  # YAML configuration
│   │   └── json_repository.py     # JSON result storage
│   │
│   ├── use_cases/           # Core ATTS algorithms
│   │   ├── atts_workflow.py          # Main 6-stage pipeline
│   │   ├── estimate_difficulty.py    # Pass@k difficulty estimation
│   │   ├── solve_problem.py          # Mode-specific generation
│   │   ├── verify_solution.py        # USVA verification
│   │   └── dialectical_refinement.py # Critic-refiner loop
│   │
│   ├── explainability/      # XAI features
│   │   ├── decision_explainer.py   # Human-readable explanations
│   │   ├── workflow_visualizer.py  # Rich console output
│   │   └── analysis_reporter.py    # Pareto analysis
│   │
│   └── interfaces/
│       └── cli.py           # Command-line interface
│
├── tests/
│   └── test_domain.py       # Unit tests
│
├── data/
│   └── math_problems.json   # Dataset (MATH problems)
│
├── results/                 # Experiment outputs (JSON)
│
└── convert_math_dataset.py  # Dataset conversion utility
```

## Configuration

### Thresholds (`config/thresholds.yaml`)

```yaml
difficulty:
  direct_threshold: 4    # d < 4 → DIRECT mode
  thinking_threshold: 7  # d < 7 → THINKING mode
  passk_k: 2             # Samples for difficulty estimation

escalation:
  threshold: 0.80        # Escalate if verification score < 0.80
  ascot_trigger: 0.60    # Immediate escalation for very low scores

refinement:
  max_iterations: 2      # Maximum dialectical refinement cycles
  early_exit_score: 0.85 # Stop early if score exceeds this
```

### CLI Options

```
--model, -m           Ollama model (default: qwen2.5:3b-instruct)
--max-problems, -n    Number of problems to process
--quick-test          Run on 5 problems only
--enable-refinement   Enable dialectical refinement (uses more tokens)
--no-escalation       Disable mode escalation
--passk-k             Pass@k samples for difficulty estimation (default: 2)
--verbose, -v         Show detailed output per problem
--skip-baseline       Skip baseline comparison
--no-rich             Disable rich console formatting
```

## Experimental Design

### Hypothesis

> ATTS achieves >20% token savings compared to always-deep baseline
> while maintaining accuracy within 5%.

### Independent Variables

- Difficulty estimation method (Pass@k sampling)
- Mode selection thresholds (τ_direct, τ_thinking)
- Escalation trigger threshold (τ_escalation)
- Refinement iterations (max_iterations)

### Dependent Variables

- **Accuracy**: Percentage of correct answers
- **Token Efficiency**: (baseline_tokens - atts_tokens) / baseline_tokens
- **Escalation Rate**: Percentage of problems requiring escalation
- **Mode Distribution**: Allocation across DIRECT/THINKING/DEEP

### Pareto Efficiency Criterion

```python
is_pareto_improvement = (token_savings > 0.20) and (accuracy_loss < 0.05)
```

## Results Format

Results are saved as JSON in `results/`:

```json
{
  "metadata": {
    "model": "qwen2.5:3b-instruct",
    "dataset_size": 100,
    "escalation_enabled": true,
    "refinement_enabled": false,
    "analysis": {
      "accuracy": {"atts": 0.94, "baseline": 0.96, "difference": -0.02},
      "tokens": {"atts_avg": 672, "baseline_avg": 847, "savings": 0.207},
      "pareto": {"is_improvement": true, "efficiency_gain": 0.35}
    }
  },
  "results": [
    {
      "id": "1",
      "true_difficulty": "easy",
      "predicted_difficulty": 3,
      "difficulty_uncertainty": 0.5,
      "initial_mode": "direct",
      "final_mode": "direct",
      "escalated": false,
      "verification_score": 0.85,
      "rubric_scores": {"LC": 0.9, "FC": 0.8, "CM": 0.85, "GA": 0.85},
      "tokens": 142,
      "correct": true,
      "decision_trace": [...]
    }
  ]
}
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_domain.py -v
```

## Dataset

The default dataset uses MATH problems. To convert from HuggingFace:

```bash
# Convert 100 problems from MATH dataset
python convert_math_dataset.py --size 100

# Creates: data/math_problems.json
```

## Hardware Requirements

| Configuration | Specs | Notes |
|--------------|-------|-------|
| Minimum | 8GB RAM, 4 CPU cores | CPU-only, slower |
| Recommended | 16GB RAM, RTX 3050+ (8GB VRAM) | Good performance |
| Tested | RTX 2050 (4GB), Ryzen 7, 16GB RAM | Works well |

### Safety Features

- Auto-checkpointing every 10 problems
- Safety breaks every 25 problems (configurable)
- Graceful interrupt handling (Ctrl+C saves progress)
- GPU temperature awareness

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if container is running
docker ps | grep ollama

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
python run_atts.py --model qwen2.5:1.5b-instruct
```

## License

MIT License - See [LICENSE](LICENSE)
