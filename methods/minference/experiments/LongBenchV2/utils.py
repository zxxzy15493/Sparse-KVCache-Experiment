import json


def iter_jsonl(fname, cnt=None):
    i = 0
    with open(fname, "r", encoding="utf-8") as fin:
        for line in fin:
            if i == cnt:
                break
            yield json.loads(line)
            i += 1


def load_data(fname):
    return list(iter_jsonl(fname))
