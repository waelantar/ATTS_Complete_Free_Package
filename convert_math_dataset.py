"""
MATH Dataset Converter - LAPTOP SAFE VERSION
Converts MATH dataset to ATTS format with safety features

Safety Features:
- Automatic subsampling to avoid overload
- Progress tracking
- Memory-efficient processing
- Clear size warnings

Usage:
    # Quick test (100 problems)
    python convert_math_dataset.py --size 100

    # Medium run (500 problems)
    python convert_math_dataset.py --size 500

    # Full dataset (use with caution on laptop!)
    python convert_math_dataset.py --size 12500
"""

import json
import argparse
from typing import List, Dict
import re

def clean_latex_answer(answer: str) -> str:
    """Extract clean answer from LaTeX boxed format"""
    # Try to extract from \boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
    if boxed_match:
        return boxed_match.group(1)

    # Fallback: return as-is
    return answer.strip()

def generate_synthetic_problems(max_size: int) -> List[Dict]:
    """Generate synthetic math problems when dataset is unavailable."""
    import random

    problems = []
    target_per_difficulty = max_size // 3

    # Easy problems (arithmetic)
    for i in range(target_per_difficulty):
        a, b = random.randint(10, 99), random.randint(10, 99)
        op = random.choice(['+', '-', '*'])
        if op == '+':
            answer = a + b
        elif op == '-':
            answer = a - b
        else:
            a, b = random.randint(2, 12), random.randint(2, 12)
            answer = a * b
        problems.append({
            "id": len(problems) + 1,
            "difficulty_label": "easy",
            "problem": f"Calculate: {a} {op} {b}",
            "answer": str(answer),
            "subject": "arithmetic",
            "original_level": 1
        })

    # Medium problems (algebra, multi-step)
    for i in range(target_per_difficulty):
        problem_type = random.choice(['linear', 'percentage', 'ratio'])
        if problem_type == 'linear':
            a = random.randint(2, 10)
            b = random.randint(1, 20)
            x = random.randint(1, 15)
            result = a * x + b
            problems.append({
                "id": len(problems) + 1,
                "difficulty_label": "medium",
                "problem": f"Solve for x: {a}x + {b} = {result}",
                "answer": str(x),
                "subject": "algebra",
                "original_level": 3
            })
        elif problem_type == 'percentage':
            base = random.randint(50, 200)
            pct = random.choice([10, 15, 20, 25, 30])
            answer = base * pct // 100
            problems.append({
                "id": len(problems) + 1,
                "difficulty_label": "medium",
                "problem": f"What is {pct}% of {base}?",
                "answer": str(answer),
                "subject": "arithmetic",
                "original_level": 3
            })
        else:
            a, b = random.randint(2, 8), random.randint(2, 8)
            total = random.randint(50, 200)
            part = total * a // (a + b)
            problems.append({
                "id": len(problems) + 1,
                "difficulty_label": "medium",
                "problem": f"Divide {total} in the ratio {a}:{b}. What is the larger part?",
                "answer": str(max(part, total - part)),
                "subject": "ratio",
                "original_level": 3
            })

    # Hard problems (quadratic, sequences, geometry)
    for i in range(target_per_difficulty):
        problem_type = random.choice(['quadratic', 'sequence', 'geometry'])
        if problem_type == 'quadratic':
            r1, r2 = random.randint(1, 8), random.randint(1, 8)
            a, b, c = 1, -(r1 + r2), r1 * r2
            problems.append({
                "id": len(problems) + 1,
                "difficulty_label": "hard",
                "problem": f"Find the sum of the roots of x² + {b}x + {c} = 0",
                "answer": str(r1 + r2),
                "subject": "algebra",
                "original_level": 5
            })
        elif problem_type == 'sequence':
            a1 = random.randint(1, 10)
            d = random.randint(2, 5)
            n = random.randint(10, 20)
            an = a1 + (n - 1) * d
            problems.append({
                "id": len(problems) + 1,
                "difficulty_label": "hard",
                "problem": f"In an arithmetic sequence, the first term is {a1} and the common difference is {d}. Find the {n}th term.",
                "answer": str(an),
                "subject": "sequences",
                "original_level": 5
            })
        else:
            r = random.randint(3, 10)
            area = r * r * 314 // 100  # Approximate pi as 3.14
            problems.append({
                "id": len(problems) + 1,
                "difficulty_label": "hard",
                "problem": f"A circle has radius {r}. Find its area (use π ≈ 3.14, round to nearest integer).",
                "answer": str(area),
                "subject": "geometry",
                "original_level": 5
            })

    # Fill remaining if needed
    while len(problems) < max_size:
        a, b = random.randint(100, 999), random.randint(10, 99)
        problems.append({
            "id": len(problems) + 1,
            "difficulty_label": "medium",
            "problem": f"Calculate: {a} + {b}",
            "answer": str(a + b),
            "subject": "arithmetic",
            "original_level": 2
        })

    random.shuffle(problems)
    for i, p in enumerate(problems):
        p["id"] = i + 1

    return problems[:max_size]


def map_difficulty(level: int) -> str:
    """
    Map MATH difficulty levels (1-5) to ATTS categories

    1-2: Easy (basic algebra, simple arithmetic)
    3-4: Medium (multi-step problems)
    5: Hard (competition-level)
    """
    if level <= 2:
        return "easy"
    elif level <= 4:
        return "medium"
    else:
        return "hard"

def convert_math_dataset(max_size: int = 100, balance_difficulties: bool = True) -> List[Dict]:
    """
    Convert MATH dataset to ATTS format

    Args:
        max_size: Maximum number of problems (laptop safe default: 100)
        balance_difficulties: Ensure even split across easy/medium/hard

    Returns:
        List of problem dictionaries
    """
    print(f"\n📦 Loading MATH dataset...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: 'datasets' library not installed")
        print("   Run: pip install datasets")
        return []

    # Load dataset - try multiple sources
    ds = None
    dataset_sources = [
        ("lighteval/MATH", "train"),
        ("competition_math", "train"),
        ("hendrycks/competition_math", "train"),
    ]

    for dataset_name, split in dataset_sources:
        try:
            print(f"   Trying: {dataset_name}...")
            ds = load_dataset(dataset_name, split=split)
            print(f"✅ Loaded {len(ds)} problems from {dataset_name}")
            break
        except Exception as e:
            print(f"   ⚠️ {dataset_name} failed: {str(e)[:50]}")
            continue

    if ds is None:
        print("❌ Could not load any MATH dataset source")
        print("   Generating synthetic math problems instead...")
        return generate_synthetic_problems(max_size)

    # Convert to ATTS format
    problems = []
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    target_per_difficulty = max_size // 3 if balance_difficulties else max_size

    for i, item in enumerate(ds):
        if len(problems) >= max_size:
            break

        difficulty = map_difficulty(item['level'])

        # Skip if we have enough of this difficulty (when balancing)
        if balance_difficulties and difficulty_counts[difficulty] >= target_per_difficulty:
            continue

        # Clean answer
        answer = clean_latex_answer(item['solution'])

        problems.append({
            "id": len(problems) + 1,
            "difficulty_label": difficulty,
            "problem": item['problem'],
            "answer": answer,
            "subject": item.get('type', 'unknown'),  # Bonus: track subject
            "original_level": item['level']  # Keep original for analysis
        })

        difficulty_counts[difficulty] += 1

    return problems

def main():
    parser = argparse.ArgumentParser(
        description="Convert MATH dataset to ATTS format (LAPTOP SAFE)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Number of problems to convert (default: 100 for safety)"
    )
    parser.add_argument(
        "--output",
        default="data/math_problems.json",
        help="Output file path"
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Don't balance difficulties (faster conversion)"
    )
    args = parser.parse_args()

    # Safety warnings
    if args.size > 1000:
        print("\n⚠️  WARNING: Large dataset selected!")
        print(f"   Requesting {args.size} problems will take significant time.")
        print(f"   Estimated runtime on RTX 2050: {args.size * 0.5 / 60:.1f} minutes")
        response = input("   Continue? (y/N): ")
        if response.lower() != 'y':
            print("   Cancelled. Use --size 100 for quick test.")
            return

    print(f"\n🔄 Converting MATH dataset...")
    print(f"   Target size: {args.size}")
    print(f"   Balance difficulties: {not args.no_balance}")

    # Convert
    problems = convert_math_dataset(
        max_size=args.size,
        balance_difficulties=not args.no_balance
    )

    if not problems:
        print("❌ No problems converted. Check errors above.")
        return

    # Show statistics
    difficulty_counts = {}
    for p in problems:
        diff = p["difficulty_label"]
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

    print(f"\n✅ Converted {len(problems)} problems:")
    for diff in ["easy", "medium", "hard"]:
        count = difficulty_counts.get(diff, 0)
        pct = count / len(problems) * 100
        print(f"   • {diff.capitalize()}: {count} ({pct:.1f}%)")

    # Save
    with open(args.output, 'w') as f:
        json.dump({"problems": problems}, f, indent=2)

    print(f"\n💾 Saved to: {args.output}")
    print(f"\n🚀 Ready to run experiment:")
    print(f"   python atts_experiment_local.py --dataset {args.output}")

    # Safety reminder
    if args.size >= 100:
        print(f"\n⚠️  LAPTOP SAFETY TIPS:")
        print(f"   • Start with --no-refinement for faster first run")
        print(f"   • Monitor temperature: nvidia-smi")
        print(f"   • Take breaks between runs")
        print(f"   • Use --max-problems to test smaller subsets first")

if __name__ == "__main__":
    main()
