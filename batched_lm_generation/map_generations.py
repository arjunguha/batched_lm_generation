import gzip
import json
from pathlib import Path
from typing import Callable
import os

TRIM_LEN = len(".json.gz")

def map_generations(p: Path, f: Callable[[str], str]):
    for file in p.glob("*.json.gz"):
        if file.name.endswith(".results.json.gz"):
            continue
        results_file = file.with_name(file.name[:-TRIM_LEN] + ".results.json.gz")
        if results_file.exists():
            continue

        results = []
        with gzip.open(file, "rt", encoding="utf-8") as infile:
            data = json.load(infile)
            for completion in data.get("completions", []):
                modified_text = f(completion["text"])
                results.append({"count": completion["count"], "text": modified_text})
        del data["completions"]
        data["results"] = results

        with gzip.open(results_file, 'wt', encoding='utf-8') as outfile:
            json.dump(data, outfile)
