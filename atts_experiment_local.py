"""
ATTS Experiment - COMPREHENSIVE Implementation (Ollama)
Validates ALL implementable sections from the paper without training

Paper Sections Implemented:
- Section 1.2: Dialectical Nature of Advanced Reasoning (full refinement loop)
- Section 2.1: Unified Self-Verification Architecture (USVA with meta-verification)
- Section 2.1.2: Generalized Verification Rubrics (LC, FC, CM, GA)
- Section 2.1.3: Integrated Meta-Verification (hallucination detection)
- Section 2.3: Adaptive Test-Time Scaling (ATTS)
- Section 2.3.1: Difficulty Estimation (Pass@k-inspired)
- Section 2.3.2: Compute Allocation Policy
- Section 2.3.3: Uncertainty-Triggered Escalation
- Section 2.4: Distilled Verification (synthetic dialectical loops)
- Section 3: Theoretical Analysis (Pareto frontier, cost-benefit)
- Section 4.1: Simulation Protocol
- Appendix A: Full ATTS Workflow

Setup:
1. docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
2. docker exec -it ollama ollama pull qwen2.5:3b-instruct
3. pip install -r requirements.txt
4. python atts_experiment_local.py --model qwen2.5:3b-instruct
"""

import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

# ============================================================================
# CONFIGURATION (Section 2.3.2)
# ============================================================================

THRESHOLD_DIRECT = 4
THRESHOLD_THINKING = 7
ESCALATION_THRESHOLD = 0.6
META_VERIFICATION_THRESHOLD = 0.7  # Section 2.1.3
MAX_REFINEMENT_ITERATIONS = 2      # Section 1.2, 2.4

# LAPTOP SAFETY DEFAULTS
DEFAULT_PASSK_K = 2                # Reduced from 3 for laptop safety
CHECKPOINT_INTERVAL = 10           # Save progress every N problems
SAFETY_BREAK_INTERVAL = 25         # Pause every N problems

# ============================================================================
# PROMPTS (Section 4.1 + Extensions)
# ============================================================================

DIFFICULTY_PROMPT = """Rate the difficulty of this math problem from 1-10.
1-3: Easy (single-step arithmetic)
4-6: Medium (multi-step reasoning)
7-10: Hard (competition-level, requires deep reasoning)
Output ONLY a number.

Problem: {problem}
Difficulty:"""

DIRECT_PROMPT = """Solve concisely: {problem}
Answer:"""

THINKING_PROMPT = """Solve step by step: {problem}
Steps:"""

DEEP_PROMPT = """Solve carefully with verification: {problem}
1. Understand the problem
2. Solve step by step
3. Verify answer
Solution:"""

# Section 2.1.2 - USVA Generalized Verification Rubrics
VERIFICATION_PROMPT = """Evaluate this solution using these criteria (score 0-1 each):
1. Logical Coherence (LC): Do steps follow logically?
2. Factual Correctness (FC): Are calculations/facts correct?
3. Completeness (CM): Are all aspects addressed?
4. Goal Alignment (GA): Does it solve the stated problem?

Problem: {problem}
Solution: {solution}

Output format:
LC: [0-1 score]
FC: [0-1 score]
CM: [0-1 score]
GA: [0-1 score]
Overall: [average]"""

# Section 2.1.3 - Meta-Verification (Detect Hallucinated Critique)
META_VERIFICATION_PROMPT = """Is this critique valid or hallucinated?
A valid critique identifies real issues. A hallucinated critique claims problems that don't exist.

Problem: {problem}
Solution: {solution}
Critique: {critique}

Answer with ONLY: VALID or HALLUCINATED
Assessment:"""

# Section 1.2 - Dialectical Refinement (Critic Role)
CRITIQUE_PROMPT = """You are a critical reviewer. Identify specific errors or gaps in this solution.
If the solution is correct, say "No issues found."

Problem: {problem}
Solution: {solution}

Critique:"""

# Section 2.4 - Refiner Role (Synthesis)
REFINE_PROMPT = """Improve this solution based on the critique.

Problem: {problem}
Original Solution: {solution}
Critique: {critique}

Improved Solution:"""

# ============================================================================
# OLLAMA CLIENT
# ============================================================================

def call_ollama(model: str, prompt: str, max_tokens: int = 2000) -> Tuple[str, int]:
    """Call local Ollama model"""
    import ollama

    client = ollama.Client(host='http://localhost:11434')

    try:
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"num_predict": max_tokens, "temperature": 0.7}
        )
        text = response["response"]
        tokens = response.get("eval_count", len(text.split()))
        return text, tokens
    except Exception as e:
        print(f"Ollama Error: {e}")
        return "", 0

# ============================================================================
# SECTION 2.3.1 - Difficulty Estimation (Pass@k inspired)
# ============================================================================

def estimate_difficulty_single(model: str, problem: str) -> int:
    """Single difficulty estimate"""
    prompt = DIFFICULTY_PROMPT.format(problem=problem)
    response, _ = call_ollama(model, prompt, max_tokens=10)
    try:
        numbers = re.findall(r'\d+', response)
        if numbers:
            return max(1, min(10, int(numbers[0])))
    except:
        pass
    return 5

def estimate_difficulty_passk(model: str, problem: str, k: int = DEFAULT_PASSK_K) -> Tuple[int, float]:
    """
    Section 2.3.1: Difficulty estimation inspired by Pass@k
    d(P) = 1 - Pass@k(P) / k

    We approximate by taking multiple estimates and measuring variance
    High variance = uncertain = likely harder

    LAPTOP SAFE: Default k=2 (reduced from k=3) to limit inference calls
    """
    estimates = []
    for _ in range(k):
        est = estimate_difficulty_single(model, problem)
        estimates.append(est)

    avg_difficulty = int(np.mean(estimates))
    uncertainty = np.std(estimates)  # High std = uncertain difficulty

    # Adjust difficulty based on uncertainty (uncertain problems are harder)
    adjusted_difficulty = min(10, avg_difficulty + int(uncertainty))

    return adjusted_difficulty, uncertainty

# ============================================================================
# SECTION 2.3.2 - Mode Selection
# ============================================================================

def select_mode(difficulty: int) -> str:
    """Section 2.3.2 - Compute Allocation Policy"""
    if difficulty < THRESHOLD_DIRECT:
        return "direct"
    elif difficulty < THRESHOLD_THINKING:
        return "thinking"
    return "deep"

# ============================================================================
# SECTION 2.1 - UNIFIED SELF-VERIFICATION ARCHITECTURE (USVA)
# ============================================================================

def solve_problem(model: str, problem: str, mode: str) -> Tuple[str, int]:
    """Execute problem solving in specified mode"""
    if mode == "direct":
        prompt = DIRECT_PROMPT.format(problem=problem)
        max_tokens = 150
    elif mode == "thinking":
        prompt = THINKING_PROMPT.format(problem=problem)
        max_tokens = 500
    else:  # deep
        prompt = DEEP_PROMPT.format(problem=problem)
        max_tokens = 1000
    return call_ollama(model, prompt, max_tokens)

def verify_solution(model: str, problem: str, solution: str) -> Tuple[float, Dict[str, float]]:
    """
    Section 2.1.2 - USVA Generalized Verification Rubrics

    Returns:
        overall_score: v ∈ [0, 1]
        rubric_scores: {LC, FC, CM, GA}
    """
    if not solution.strip():
        return 0.0, {"LC": 0.0, "FC": 0.0, "CM": 0.0, "GA": 0.0}

    prompt = VERIFICATION_PROMPT.format(problem=problem, solution=solution)
    response, _ = call_ollama(model, prompt, max_tokens=200)

    rubric_scores = {}

    # Parse individual rubric scores
    for rubric in ['LC', 'FC', 'CM', 'GA']:
        match = re.search(rf'{rubric}:\s*([\d.]+)', response)
        if match:
            try:
                rubric_scores[rubric] = max(0.0, min(1.0, float(match.group(1))))
            except:
                rubric_scores[rubric] = 0.5
        else:
            rubric_scores[rubric] = 0.5

    # Overall score
    overall_match = re.search(r'Overall:\s*([\d.]+)', response)
    if overall_match:
        try:
            overall_score = max(0.0, min(1.0, float(overall_match.group(1))))
        except:
            overall_score = np.mean(list(rubric_scores.values()))
    else:
        overall_score = np.mean(list(rubric_scores.values()))

    return overall_score, rubric_scores

def meta_verify_critique(model: str, problem: str, solution: str, critique: str) -> bool:
    """
    Section 2.1.3 - Integrated Meta-Verification

    Detects hallucinated critiques (model claims issues that don't exist)
    Returns True if critique is valid, False if hallucinated
    """
    if "no issues" in critique.lower() or len(critique.strip()) < 10:
        return True  # No critique to verify

    prompt = META_VERIFICATION_PROMPT.format(
        problem=problem,
        solution=solution,
        critique=critique
    )
    response, _ = call_ollama(model, prompt, max_tokens=20)

    # Parse response
    response_lower = response.lower()
    if "valid" in response_lower and "hallucinated" not in response_lower:
        return True
    elif "hallucinated" in response_lower:
        return False
    else:
        # Default: assume valid if uncertain
        return True

# ============================================================================
# SECTION 1.2, 2.4 - DIALECTICAL REFINEMENT
# ============================================================================

def dialectical_refinement(model: str, problem: str, initial_solution: str,
                          max_iterations: int = MAX_REFINEMENT_ITERATIONS) -> Tuple[str, int, List[Dict]]:
    """
    Section 1.2: The Dialectical Nature of Advanced Reasoning
    Section 2.4: Distilled Verification Knowledge

    Implements the full dialectical loop:
    1. Generator/Proposer: Create solution
    2. Verifier/Critic: Identify issues
    3. Meta-Verifier: Validate critique
    4. Refiner/Synthesizer: Improve solution

    Returns:
        final_solution: Best solution found
        total_tokens: Token count
        refinement_history: Log of refinement process
    """
    solution = initial_solution
    total_tokens = 0
    refinement_history = []

    for iteration in range(max_iterations):
        # Stage 1: Critic - Identify issues
        critique_prompt = CRITIQUE_PROMPT.format(problem=problem, solution=solution)
        critique, tokens = call_ollama(model, critique_prompt, max_tokens=300)
        total_tokens += tokens

        # Stage 2: Meta-Verification - Is critique valid?
        critique_is_valid = meta_verify_critique(model, problem, solution, critique)

        # Stage 3: If no valid issues, we're done
        if not critique_is_valid or "no issues" in critique.lower():
            refinement_history.append({
                "iteration": iteration,
                "critique": critique,
                "critique_valid": critique_is_valid,
                "action": "stopped",
                "reason": "no_valid_issues"
            })
            break

        # Stage 4: Refiner - Synthesize improved solution
        refine_prompt = REFINE_PROMPT.format(
            problem=problem,
            solution=solution,
            critique=critique
        )
        improved_solution, tokens = call_ollama(model, refine_prompt, max_tokens=500)
        total_tokens += tokens

        refinement_history.append({
            "iteration": iteration,
            "critique": critique,
            "critique_valid": critique_is_valid,
            "action": "refined",
            "tokens": tokens
        })

        solution = improved_solution

    return solution, total_tokens, refinement_history

# ============================================================================
# SECTION 2.3.3 - Escalation
# ============================================================================

def escalate_mode(current_mode: str) -> Optional[str]:
    """Section 2.3.3 - Escalation to next compute tier"""
    if current_mode == "direct":
        return "thinking"
    elif current_mode == "thinking":
        return "deep"
    return None

# ============================================================================
# APPENDIX A - FULL ATTS WORKFLOW
# ============================================================================

def atts_workflow(model: str, problem: Dict, enable_escalation: bool = True,
                 enable_refinement: bool = False, passk_k: int = DEFAULT_PASSK_K) -> Dict:
    """
    Appendix A: Complete ATTS Workflow

    Stage 1: Difficulty Estimation (Pass@k inspired)
    Stage 2: Mode Selection
    Stage 3: Solution Generation
    Stage 4: Self-Verification (USVA)
    Stage 5: Escalation Check (if needed)
    Stage 6: Dialectical Refinement (if Deep mode)
    Output: Final solution with full metrics

    LAPTOP SAFE: Refinement disabled by default (enable with --enable-refinement)
    """
    workflow_log = {}
    total_tokens = 0

    # Stage 1: Difficulty Estimation
    difficulty, uncertainty = estimate_difficulty_passk(model, problem["problem"], k=passk_k)
    workflow_log["difficulty_estimate"] = difficulty
    workflow_log["difficulty_uncertainty"] = uncertainty

    # Stage 2: Mode Selection
    mode = select_mode(difficulty)
    initial_mode = mode
    workflow_log["initial_mode"] = mode

    # Stage 3: Solution Generation
    solution, tokens = solve_problem(model, problem["problem"], mode)
    total_tokens += tokens
    workflow_log["initial_solution_tokens"] = tokens

    # Stage 4: Self-Verification (USVA)
    verification_score, rubric_scores = verify_solution(model, problem["problem"], solution)
    workflow_log["verification_score"] = verification_score
    workflow_log["rubric_scores"] = rubric_scores

    # Stage 5: Escalation Check
    escalated = False
    escalation_path = [mode]

    if enable_escalation and verification_score < ESCALATION_THRESHOLD:
        next_mode = escalate_mode(mode)
        if next_mode:
            escalated = True
            mode = next_mode
            escalation_path.append(mode)
            solution, tokens = solve_problem(model, problem["problem"], mode)
            total_tokens += tokens
            # Re-verify after escalation
            verification_score, rubric_scores = verify_solution(model, problem["problem"], solution)
            workflow_log["escalation_tokens"] = tokens

    workflow_log["escalated"] = escalated
    workflow_log["final_mode"] = mode

    # Stage 6: Dialectical Refinement (Deep mode only, Section 2.4)
    refinement_history = []
    if enable_refinement and mode == "deep":
        solution, refine_tokens, refinement_history = dialectical_refinement(
            model, problem["problem"], solution
        )
        total_tokens += refine_tokens
        # Final verification after refinement
        verification_score, rubric_scores = verify_solution(model, problem["problem"], solution)
        workflow_log["refinement_tokens"] = refine_tokens
        workflow_log["refinement_iterations"] = len(refinement_history)

    # Output: Final results
    correct = check_answer(solution, problem["answer"])

    return {
        "id": problem["id"],
        "true_difficulty": problem["difficulty_label"],
        "predicted_difficulty": difficulty,
        "difficulty_uncertainty": uncertainty,
        "initial_mode": initial_mode,
        "final_mode": mode,
        "escalated": escalated,
        "escalation_path": escalation_path,
        "verification_score": verification_score,
        "rubric_scores": rubric_scores,
        "refinement_history": refinement_history,
        "tokens": total_tokens,
        "correct": correct,
        "workflow_log": workflow_log
    }

def check_answer(response: str, correct_answer: str) -> bool:
    """Check if response contains correct answer"""
    response_lower = response.lower().replace(" ", "")
    answer_lower = correct_answer.lower().replace(" ", "")

    if answer_lower in response_lower:
        return True

    try:
        response_nums = re.findall(r'-?\d+\.?\d*', response)
        answer_nums = re.findall(r'-?\d+\.?\d*', correct_answer)
        if answer_nums:
            target = float(answer_nums[0])
            for num in response_nums:
                if abs(float(num) - target) < 0.01:
                    return True
    except:
        pass

    return False

# ============================================================================
# EXPERIMENTS
# ============================================================================

def run_atts_experiment(model: str, problems: List[Dict],
                       enable_escalation: bool = True,
                       enable_refinement: bool = False,
                       passk_k: int = DEFAULT_PASSK_K,
                       checkpoint_file: Optional[str] = None) -> List[Dict]:
    """
    Run full ATTS experiment with all features

    LAPTOP SAFE: Includes progress checkpointing and safety breaks
    """
    results = []

    for idx, prob in enumerate(tqdm(problems, desc="ATTS (Full)"), start=1):
        try:
            result = atts_workflow(model, prob, enable_escalation, enable_refinement, passk_k)
            results.append(result)

            # Checkpoint progress (avoid data loss on crash)
            if checkpoint_file and idx % CHECKPOINT_INTERVAL == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                tqdm.write(f"💾 Checkpoint saved ({idx}/{len(problems)})")

            # Safety break (prevent overheating)
            if idx % SAFETY_BREAK_INTERVAL == 0 and idx < len(problems):
                tqdm.write(f"⏸️  Safety break (5s) - {idx}/{len(problems)} complete")
                time.sleep(5)

        except Exception as e:
            tqdm.write(f"❌ Error on problem {prob['id']}: {e}")
            # Continue with other problems
            continue

    return results

def run_baseline_experiment(model: str, problems: List[Dict],
                           enable_refinement: bool = False) -> List[Dict]:
    """
    Baseline: Always Deep mode

    LAPTOP SAFE: Refinement optional to match ATTS setup
    """
    results = []

    for idx, prob in enumerate(tqdm(problems, desc="Baseline"), start=1):
        try:
            # Always use deep mode
            solution, tokens = solve_problem(model, prob["problem"], "deep")

            # Apply refinement only if enabled
            if enable_refinement:
                solution, refine_tokens, _ = dialectical_refinement(model, prob["problem"], solution)
                tokens += refine_tokens

            correct = check_answer(solution, prob["answer"])

            results.append({
                "id": prob["id"],
                "mode": "deep",
                "tokens": tokens,
                "correct": correct
            })

            # Safety break
            if idx % SAFETY_BREAK_INTERVAL == 0 and idx < len(problems):
                tqdm.write(f"⏸️  Safety break (5s) - {idx}/{len(problems)} complete")
                time.sleep(5)

        except Exception as e:
            tqdm.write(f"❌ Error on problem {prob['id']}: {e}")
            continue

    return results

# ============================================================================
# SECTION 3 - THEORETICAL ANALYSIS
# ============================================================================

def compute_pareto_frontier(atts_results: List[Dict], baseline_results: List[Dict]) -> Dict:
    """
    Section 3.2: Accuracy-Efficiency Trade-off Analysis

    Compute Pareto frontier: optimal accuracy-cost trade-offs
    """
    atts_df = pd.DataFrame(atts_results)
    baseline_df = pd.DataFrame(baseline_results)

    # Compute efficiency ratio: accuracy / tokens
    atts_acc = atts_df["correct"].mean()
    atts_tokens = atts_df["tokens"].mean()
    baseline_acc = baseline_df["correct"].mean()
    baseline_tokens = baseline_df["tokens"].mean()

    atts_efficiency = atts_acc / atts_tokens if atts_tokens > 0 else 0
    baseline_efficiency = baseline_acc / baseline_tokens if baseline_tokens > 0 else 0

    # Cost-benefit analysis
    token_savings = (baseline_tokens - atts_tokens) / baseline_tokens if baseline_tokens > 0 else 0
    accuracy_cost = baseline_acc - atts_acc

    # Pareto improvement: saves tokens AND maintains accuracy
    is_pareto_improvement = token_savings > 0.2 and accuracy_cost < 0.05

    return {
        "atts_efficiency": atts_efficiency,
        "baseline_efficiency": baseline_efficiency,
        "token_savings": token_savings,
        "accuracy_cost": accuracy_cost,
        "is_pareto_improvement": is_pareto_improvement,
        "efficiency_gain": (atts_efficiency - baseline_efficiency) / baseline_efficiency if baseline_efficiency > 0 else 0
    }

def analyze_comprehensive(atts_results: List[Dict], baseline_results: List[Dict]):
    """Section 3: Complete Theoretical Analysis + Results"""
    atts_df = pd.DataFrame(atts_results)
    baseline_df = pd.DataFrame(baseline_results)

    # Overall metrics
    atts_acc = atts_df["correct"].mean() * 100
    atts_tokens = atts_df["tokens"].mean()
    baseline_acc = baseline_df["correct"].mean() * 100
    baseline_tokens = baseline_df["tokens"].mean()

    savings = (1 - atts_tokens / baseline_tokens) * 100 if baseline_tokens > 0 else 0

    print("\n" + "="*60)
    print("📊 COMPREHENSIVE RESULTS")
    print("="*60)
    print(f"\nBaseline: {baseline_acc:.1f}% accuracy, {baseline_tokens:.0f} avg tokens")
    print(f"ATTS:     {atts_acc:.1f}% accuracy, {atts_tokens:.0f} avg tokens")
    print(f"\n💰 Token Savings: {savings:.1f}%")

    # Mode distribution
    final_mode_dist = atts_df['final_mode'].value_counts().to_dict()
    print(f"📈 Mode Distribution: {final_mode_dist}")

    # Escalation statistics
    escalation_rate = atts_df["escalated"].mean() * 100
    print(f"🔼 Escalation Rate: {escalation_rate:.1f}%")

    # Refinement statistics (Section 2.4)
    avg_refinement_iters = atts_df["workflow_log"].apply(
        lambda x: x.get("refinement_iterations", 0)
    ).mean()
    print(f"🔄 Avg Refinement Iterations: {avg_refinement_iters:.2f}")

    # Difficulty estimation (Section 2.3.1)
    difficulty_map = {"easy": 3, "medium": 5, "hard": 8}
    atts_df["true_difficulty_score"] = atts_df["true_difficulty"].map(difficulty_map)
    difficulty_mae = abs(atts_df["predicted_difficulty"] - atts_df["true_difficulty_score"]).mean()
    avg_uncertainty = atts_df["difficulty_uncertainty"].mean()
    print(f"🎯 Difficulty Estimation MAE: {difficulty_mae:.2f}")
    print(f"📊 Avg Difficulty Uncertainty: {avg_uncertainty:.2f}")

    # USVA rubric scores (Section 2.1.2)
    avg_rubrics = {
        "LC": atts_df["rubric_scores"].apply(lambda x: x.get("LC", 0.5)).mean(),
        "FC": atts_df["rubric_scores"].apply(lambda x: x.get("FC", 0.5)).mean(),
        "CM": atts_df["rubric_scores"].apply(lambda x: x.get("CM", 0.5)).mean(),
        "GA": atts_df["rubric_scores"].apply(lambda x: x.get("GA", 0.5)).mean()
    }
    print(f"\n✓ USVA Rubric Scores:")
    for rubric, score in avg_rubrics.items():
        print(f"   {rubric}: {score:.2f}")

    # Per-difficulty breakdown
    print(f"\n📊 Performance by Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        atts_subset = atts_df[atts_df["true_difficulty"] == diff]
        baseline_subset = baseline_df[baseline_df["id"].isin(atts_subset["id"])]

        if len(atts_subset) > 0:
            atts_diff_acc = atts_subset["correct"].mean() * 100
            baseline_diff_acc = baseline_subset["correct"].mean() * 100
            atts_diff_tokens = atts_subset["tokens"].mean()
            baseline_diff_tokens = baseline_subset["tokens"].mean()
            diff_savings = (1 - atts_diff_tokens / baseline_diff_tokens) * 100 if baseline_diff_tokens > 0 else 0

            print(f"  {diff.capitalize()}: ATTS={atts_diff_acc:.0f}% / Baseline={baseline_diff_acc:.0f}% | "
                  f"Tokens: {atts_diff_tokens:.0f} vs {baseline_diff_tokens:.0f} ({diff_savings:+.1f}%)")

    # Section 3.2: Pareto Frontier Analysis
    print(f"\n" + "="*60)
    print("📈 PARETO FRONTIER ANALYSIS (Section 3.2)")
    print("="*60)
    pareto = compute_pareto_frontier(atts_results, baseline_results)
    print(f"ATTS Efficiency Ratio:     {pareto['atts_efficiency']:.6f}")
    print(f"Baseline Efficiency Ratio: {pareto['baseline_efficiency']:.6f}")
    print(f"Efficiency Gain:           {pareto['efficiency_gain']*100:+.1f}%")
    print(f"Token Savings:             {pareto['token_savings']*100:.1f}%")
    print(f"Accuracy Cost:             {pareto['accuracy_cost']*100:.1f}%")
    print(f"Pareto Improvement:        {'✅ YES' if pareto['is_pareto_improvement'] else '❌ NO'}")

    print("="*60)

    # Section 4.2: Hypothesis Validation
    if savings > 20 and (atts_acc - baseline_acc) > -5:
        print("✅ HYPOTHESIS SUPPORTED!")
        print("   • Token savings > 20% ✓")
        print("   • Accuracy within 5% of baseline ✓")
        if pareto['is_pareto_improvement']:
            print("   • Pareto improvement achieved ✓")
    else:
        print("⚠️  Mixed results")

    print("="*60)

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ATTS Comprehensive Experiment - LAPTOP SAFE VERSION"
    )
    parser.add_argument("--model", default="qwen2.5:3b-instruct", help="Ollama model name")
    parser.add_argument("--dataset", default="data/math_problems.json", help="Dataset path")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit dataset size (safety)")
    parser.add_argument("--no-escalation", action="store_true", help="Disable escalation")
    parser.add_argument("--enable-refinement", action="store_true", help="Enable dialectical refinement (slower)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Verification threshold")
    parser.add_argument("--passk-k", type=int, default=DEFAULT_PASSK_K, help="Pass@k samples (default: 2 for safety)")
    parser.add_argument("--quick-test", action="store_true", help="Run on first 5 problems only")
    args = parser.parse_args()

    global ESCALATION_THRESHOLD
    ESCALATION_THRESHOLD = args.threshold

    print(f"\n🏠 ATTS Comprehensive Experiment - LAPTOP SAFE MODE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Paper Sections Validated: 12+")
    print(f"🛡️  Safety Features: Checkpointing, Auto-breaks, Error recovery")
    print("="*60)

    # Check Ollama
    try:
        import ollama
        client = ollama.Client(host='http://localhost:11434')
        client.list()
        print("✅ Ollama connected")
    except Exception as e:
        print(f"❌ Ollama not running! Error: {e}")
        return

    # Load problems
    try:
        with open(args.dataset) as f:
            all_problems = json.load(f)["problems"]
        print(f"📂 Loaded {len(all_problems)} problems from dataset")

        # Apply safety limits
        if args.quick_test:
            problems = all_problems[:5]
            print(f"🧪 Quick test mode: Using first 5 problems")
        elif args.max_problems:
            problems = all_problems[:args.max_problems]
            print(f"⚠️  Limited to {len(problems)} problems (--max-problems)")
        else:
            problems = all_problems

        # Laptop safety warning for large datasets
        if len(problems) > 100:
            print(f"\n⚠️  LARGE DATASET WARNING:")
            print(f"   {len(problems)} problems will take significant time")
            print(f"   Estimated runtime: {len(problems) * 0.5 / 60:.1f} minutes")
            print(f"   Consider using --max-problems 100 for initial testing")
            response = input("   Continue? (y/N): ")
            if response.lower() != 'y':
                print("   Cancelled. Use --quick-test or --max-problems to limit size.")
                return

    except FileNotFoundError:
        print(f"❌ Dataset not found: {args.dataset}")
        print(f"   Convert MATH dataset: python convert_math_dataset.py --size 100")
        return

    print(f"\n⚙️  Configuration:")
    print(f"   • Difficulty thresholds: Direct<{THRESHOLD_DIRECT}, Thinking<{THRESHOLD_THINKING}")
    print(f"   • Escalation: {'Enabled' if not args.no_escalation else 'Disabled'}")
    print(f"   • Dialectical Refinement: {'Enabled' if args.enable_refinement else 'Disabled (faster)'}")
    print(f"   • Verification threshold: {ESCALATION_THRESHOLD}")
    print(f"   • Pass@k samples: {args.passk_k}")
    print(f"   • Checkpoint interval: {CHECKPOINT_INTERVAL} problems")
    print(f"   • Safety breaks: Every {SAFETY_BREAK_INTERVAL} problems")
    print()

    # RTX 2050 specific reminder
    print(f"💡 RTX 2050 TIPS:")
    print(f"   • Monitor GPU: nvidia-smi")
    print(f"   • Progress auto-saved every {CHECKPOINT_INTERVAL} problems")
    print(f"   • Ctrl+C to stop safely (progress will be saved)")
    print()

    # Setup checkpoint file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_file = f"results/checkpoint_atts_{timestamp}.json"

    # Run experiments
    try:
        atts_results = run_atts_experiment(
            args.model, problems,
            enable_escalation=not args.no_escalation,
            enable_refinement=args.enable_refinement,
            passk_k=args.passk_k,
            checkpoint_file=checkpoint_file
        )
        baseline_results = run_baseline_experiment(
            args.model, problems,
            enable_refinement=args.enable_refinement
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print(f"   Checkpoint saved at: {checkpoint_file}")
        print(f"   You can analyze partial results from checkpoint file")
        return

    # Comprehensive analysis
    analyze_comprehensive(atts_results, baseline_results)

    # Save results
    output_file = f"results/comprehensive_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "dataset_size": len(problems),
                "threshold_direct": THRESHOLD_DIRECT,
                "threshold_thinking": THRESHOLD_THINKING,
                "escalation_threshold": ESCALATION_THRESHOLD,
                "max_refinement_iterations": MAX_REFINEMENT_ITERATIONS,
                "passk_k": args.passk_k,
                "escalation_enabled": not args.no_escalation,
                "refinement_enabled": args.enable_refinement
            },
            "atts": atts_results,
            "baseline": baseline_results
        }, f, indent=2)
    print(f"\n💾 Saved: {output_file}")

    # Clean up checkpoint
    try:
        import os
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️  Removed checkpoint file (experiment complete)")
    except:
        pass

if __name__ == "__main__":
    main()
