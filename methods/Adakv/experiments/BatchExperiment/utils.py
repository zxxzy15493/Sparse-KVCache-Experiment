import json


def iter_jsonl(fname, cnt=None):
    with open(fname, encoding="utf-8") as handle:
        count = 0
        for line in handle:
            if not line.strip():
                continue
            if cnt is not None and count >= cnt:
                break
            yield json.loads(line)
            count += 1


def load_data(fname):
    return list(iter_jsonl(fname))
