import gzip
import json
from pathlib import Path
from typing import Callable
import os

def map_generations(p: Path, f: Callable[[str], str]):
    for file in p.glob("*.json.gz"):
        if file.name.endswith(".results.json.gz"):
            continue
        results_file = file.with_suffix('.results.json.gz')
        if results_file.exists():
            print(f"{results_file} already exists. Skipping processing.")
            continue

        results = []
        with gzip.open(file, 'rt', encoding='utf-8') as infile:
            data = json.load(infile)
            for completion in data.get("completions", []):
                modified_text = f(completion["text"])
                results.append({"count": completion["count"], "text": modified_text})

        with gzip.open(results_file, 'wt', encoding='utf-8') as outfile:
            json.dump(results, outfile)
