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

    # Load dataset (train split)
    try:
        ds = load_dataset('hendrycks/MATH', split='train', trust_remote_code=True)
        print(f"✅ Loaded {len(ds)} problems from MATH dataset")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("   Make sure you have internet connection")
        return []

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
