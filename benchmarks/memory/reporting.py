"""将 CPU 内存 CSV 渲染为等宽文本表。"""

import csv


SUMMARY_COLUMNS = (
    ("model", "Model", "<"),
    ("method", "Method", "<"),
    ("batch_size", "Batch", ">"),
    ("context_len", "Context", ">"),
    ("budget", "Budget", ">"),
    ("output_length", "Output", ">"),
    ("status", "Status", "<"),
    ("model_loading_peak_mib", "Load peak", ">"),
    ("inference_peak_mib", "Infer peak", ">"),
    ("inference_peak_increase_mib", "Peak increase", ">"),
    ("process_peak_mib", "Process peak", ">"),
)
PHASE_COLUMNS = (
    ("phase", "Phase", "<"),
    ("prefill_mode", "Prefill mode", "<"),
    ("rss_start_mib", "Start", ">"),
    ("rss_end_mib", "End", ">"),
    ("rss_peak_mib", "Peak", ">"),
    ("rss_peak_increase_mib", "Peak increase", ">"),
)
REPORTS = {
    "summary": ("CPU memory summary (MiB)", SUMMARY_COLUMNS),
    "phases": ("CPU memory by phase (MiB)", PHASE_COLUMNS),
}


def write_aligned_report(csv_path):
    title, columns = REPORTS[csv_path.stem]
    fields, headers, alignments = zip(*columns)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [[row[field] or "-" for field in fields] for row in csv.DictReader(handle)]

    widths = [max(map(len, column)) for column in zip(headers, *rows)]
    row_format = "  ".join(f"{{:{alignment}{width}}}" for alignment, width in zip(alignments, widths))
    separator = "  ".join("-" * width for width in widths)
    lines = [title, "", row_format.format(*headers), separator]
    lines.extend(row_format.format(*row) for row in rows)

    report_path = csv_path.with_suffix(".txt")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
