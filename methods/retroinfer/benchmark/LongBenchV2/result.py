import os
import json
import argparse


def compute_scores(pred_dir, compensated=False):
    """
    Read prediction files from pred_dir and compute accuracy scores.

    Returns a dict with keys: overall, easy, hard, short, medium, long, total_count
    """
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    files = [f for f in os.listdir(pred_dir) if f.endswith(('.json', '.jsonl'))]
    if not files:
        raise FileNotFoundError(f"No prediction files (.json/.jsonl) found in: {pred_dir}")

    # Aggregate all predictions
    all_preds = []
    for file in sorted(files):
        filepath = os.path.join(pred_dir, file)
        try:
            pred_data = json.load(open(filepath, encoding='utf-8'))
        except Exception:
            pred_data = [json.loads(line) for line in open(filepath, encoding='utf-8') if line.strip()]
        all_preds.extend(pred_data)

    # Compute metrics
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0.0, 0.0, 0.0, 0.0, 0.0

    for pred in all_preds:
        acc = int(pred.get('judge', 0))
        if compensated and pred.get('pred') is None:
            acc = 0.25

        difficulty = pred.get('difficulty', 'easy')
        if difficulty == 'easy':
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        length = pred.get('length', 'short')
        if length == 'short':
            short += 1
            short_acc += acc
        elif length == 'medium':
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    total = len(all_preds)

    def safe_ratio(numerator, denominator):
        return round(100 * numerator / denominator, 1) if denominator > 0 else 0.0

    scores = {
        'total': total,
        'overall': safe_ratio(easy_acc + hard_acc, total),
        'easy': safe_ratio(easy_acc, easy),
        'hard': safe_ratio(hard_acc, hard),
        'short': safe_ratio(short_acc, short),
        'medium': safe_ratio(medium_acc, medium),
        'long': safe_ratio(long_acc, long),
        'easy_count': easy,
        'hard_count': hard,
        'short_count': short,
        'medium_count': medium,
        'long_count': long,
    }
    return scores


def format_scores(scores):
    """Format scores as a tab-separated table string (matching the original output format)."""
    header = "Model\tOverall\tEasy\tHard\tShort\tMedium\tLong"
    row = (
        f"Results\t"
        f"{scores['overall']}\t"
        f"{scores['easy']}\t"
        f"{scores['hard']}\t"
        f"{scores['short']}\t"
        f"{scores['medium']}\t"
        f"{scores['long']}"
    )
    return f"{header}\n{row}"


def main():
    parser = argparse.ArgumentParser(
        description="Score LongBenchV2 prediction results."
    )
    parser.add_argument(
        "--save_dir", "-s",
        type=str,
        required=True,
        help="Directory containing prediction JSON/JSONL files (same as pred.py's --save_dir)."
    )
    parser.add_argument(
        "--compensated", "-c",
        action='store_true',
        help="If set, assign 0.25 accuracy when pred is None (random guess compensation)."
    )
    parser.add_argument(
        "--output_file", "-o",
        type=str,
        default=None,
        help="Path to save the result. Defaults to <save_dir>/result.txt."
    )
    args = parser.parse_args()

    scores = compute_scores(args.save_dir, compensated=args.compensated)

    # Build output
    output_lines = [format_scores(scores)]
    output_lines.append("")
    output_lines.append(f"Total samples: {scores['total']}")
    output_lines.append(f"Easy: {scores['easy_count']} (acc={scores['easy']}%)")
    output_lines.append(f"Hard: {scores['hard_count']} (acc={scores['hard']}%)")
    output_lines.append(f"Short: {scores['short_count']} (acc={scores['short']}%)")
    output_lines.append(f"Medium: {scores['medium_count']} (acc={scores['medium']}%)")
    output_lines.append(f"Long: {scores['long_count']} (acc={scores['long']}%)")

    output_text = '\n'.join(output_lines)

    # Print to stdout
    print(output_text)

    # Save to file
    output_file = args.output_file or os.path.join(args.save_dir, 'result.txt')
    os.makedirs(os.path.dirname(output_file) or args.save_dir, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_text)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
