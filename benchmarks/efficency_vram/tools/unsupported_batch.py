"""识别运行时明确指出不支持多 batch 的错误。"""

from __future__ import annotations

import re


UNSUPPORTED_BATCH_PATTERNS = (
    re.compile(r"only supports? batch sizes?\s*(?:greater than\s*)?1\b"),
    re.compile(r"only supports? batch size\s*1\b"),
    re.compile(r"supports? batch size\s*1\s+only\b"),
    re.compile(r"requires? batch size\s*1\b"),
    re.compile(r"(?:do|does) not support (?:batch(?: size)?|bsz)(?:\s*[><=]+\s*\d+)?"),
    re.compile(r"(?:unsupported|not supported) batch(?: size)?(?:\s*[><=]+\s*\d+)?"),
    re.compile(r"cannot handle batch sizes?\s*>\s*1\b"),
    re.compile(r"batching is not implemented"),
    re.compile(r"assert\s+(?:\w+\.)?(?:bsz|batch_size)\s*==\s*1\b"),
)


def is_unsupported_batch_error(error_text: str) -> bool:
    """仅当 traceback 明确指出 batch 限制时返回 true。"""
    normalized = error_text.lower().replace("_", " ")
    return any(pattern.search(normalized) for pattern in UNSUPPORTED_BATCH_PATTERNS)
