from .util import read_json_gz
from pathlib import Path
from typing import List, Optional
import argparse
import json
import re


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


def _all_completions(
    completions_data: dict, 
    completion_limit: Optional[int], 
    with_prompts: Optional[bool]
) -> List[str]:
    """
    Returns all the completions in completions_data.
    """
    results = []
    count = 0
    for item in completions_data["completions"]:
        for _ in range(item["count"]):
            if completion_limit is not None and count >= completion_limit:
                return results
            if with_prompts:
                results.append(completions_data["prompt"] + item["text"])
            else:
                results.append(item["text"])
            count += 1
    return results


def main_with_args(dir: Path, output_file: Path, completion_limit: int = None, with_prompts: bool = None):
    """
    Reads all the completions files in the directory dir and writes them to the output file.
    """
    files = read_completions_dir(dir)
    results = []
    for p in files:
        completions_data = read_json_gz(p)
        results.append(_all_completions(completions_data, completion_limit, with_prompts))

    with output_file.open("w") as f:
        json.dump(results, f, indent=2)


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
        "dir", type=Path, help="The directory containing the completions files."
    )
    parser.add_argument(
        "output_file", type=Path, help="The file to write the completions to."
    )
    parser.add_argument(
        "--with-prompts", action="store_true"
    )
    args = parser.parse_args()
    main_with_args(args.dir, args.output_file, args.completion_limit, args.with_prompts)


if __name__ == "__main__":
    main()
