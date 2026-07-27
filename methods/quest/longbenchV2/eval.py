import json
import argparse
from collections import defaultdict


def is_correct(item):
    """
     judge 
     judge  pred == answer
    """
    if "judge" in item:
        judge = item["judge"]

        # judge  bool
        if isinstance(judge, bool):
            return judge

        # judge  "true" / "false"
        if isinstance(judge, str):
            return judge.lower() == "true"

        return bool(judge)

    #  judge  pred  answer 
    pred = item.get("pred")
    answer = item.get("answer")
    return pred is not None and answer is not None and str(pred).strip() == str(answer).strip()


def calculate_difficulty_score(input_path, output_path):
    stats = {
        "easy": {"correct": 0, "total": 0},
        "hard": {"correct": 0, "total": 0},
    }

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            difficulty = item.get("difficulty")

            #  easy  hard
            if difficulty not in stats:
                continue

            stats[difficulty]["total"] += 1

            if is_correct(item):
                stats[difficulty]["correct"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for difficulty in ["easy", "hard"]:
            correct = stats[difficulty]["correct"]
            total = stats[difficulty]["total"]

            if total == 0:
                score = 0.0
            else:
                score = correct / total

            result = {
                "difficulty": difficulty,
                "correct": correct,
                "total": total,
                "score": f"{correct}/{total}",
                "accuracy": round(score, 4),
                "accuracy_percent": round(score * 100, 2),
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=" jsonl ")
    parser.add_argument("--output", required=True, help=" jsonl ")
    args = parser.parse_args()

    calculate_difficulty_score(args.input, args.output)