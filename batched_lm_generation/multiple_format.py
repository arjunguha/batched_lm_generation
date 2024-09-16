"""
Converts results from batched_lm_generation into the MultiPL-E completions 
format. The MultiPL-E format is what inspired this tool. The two differences
are that (1) MultiPL-E does not have the "count" field in the "completions"
array and (2) it has extra fields for "tests", "language", and "name":

{
    "prompt": PROMPT,
    "temperature": TEMP,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "stop_tokens": [ STOP_TOKEN ... ],
    "completions": [ COMPLETION ... ],
    "tests": [ STRING ... ],
    "language": STRING,
    "name": STRING
}

https://github.com/nuprl/MultiPL-E/blob/main/multipl_e/completions.py#L202

So, in this file, we "explode" the completions field to remove the count
and add the extra fields from "extras".
"""


from .util import read_json_gz
from pathlib import Path
from typing import List, Optional
import argparse
import json
import re
import gzip


def item_number(p: Path) -> int:
    """
    Returns the item number from a completions file path.
    The name is Item_N.json.gz, where N is the item number.
    """
    match = re.search(r"Item_(\d+)\.json\.gz", p.name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Invalid file name: {p.name}")


def read_completions_dir(p: Path) -> List[Path]:
    """
    Returns all the completions files in the directory p, sorted by index.

    Each file is named Item_N.json.gz, where N is the index.
    """
    files = list(p.glob("Item_*.json.gz"))
    files.sort(key=item_number)
    return files


def _transform(completions_data: dict, completion_limit: Optional[int]) -> List[str]:
    """
    Returns all the completions in completions_data.
    """
    new_completions = []
    count = 0

    for item in completions_data["completions"]:
        for _ in range(item["count"]):
            if completion_limit is not None and count >= completion_limit:
                return results
            new_completions.append(item["text"])
            count += 1
    completions_data["completions"] = new_completions

    for key, value in completions_data["extras"].items():
        completions_data[key] = value
    return completions_data


def main_with_args(input_dir: Path, output_dir: Path, completion_limit: int, tests_field: str, language: Optional[str]):
    """
    Reads all the completions files in the directory dir and writes them to the output file.
    """

    assert input_dir != output_dir
    output_dir.mkdir(exist_ok=True)

    for p in read_completions_dir(input_dir):
        completions_data = read_json_gz(p)
        output_file = output_dir / p.name
        with gzip.open(output_file, "wt") as f:
            completions_data = _transform(completions_data, completion_limit)
            if language:
                completions_data = {**completions_data, "language": language}
            json.dump({**completions_data, "tests": completions_data[tests_field]}, f)


def main():
    parser = argparse.ArgumentParser(
        description="Concatenates all the completions in a directory into a single file."
    )
    parser.add_argument(
        "--completion-limit",
        type=int,
        help="The maximum number of completions to output.",
    )
    parser.add_argument(
        "--tests-field", type=str, default="tests",
        help="Which field is to be used as tests in Multipl-E."
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Optional language for Multipl-E. If not specified, assumes language field is already in extras."
    )
    parser.add_argument(
        "input_dir", type=Path, help="The directory containing the completions files."
    )
    parser.add_argument(
        "output_dir", type=Path, help="The file to write the completions to."
    )
    args = parser.parse_args()
    main_with_args(**vars(args))


if __name__ == "__main__":
    main()
