import subprocess
import pytest
from pathlib import Path
from ..map_generations import map_generations

def test_automodel_base_generation_and_map():
    TEST_DIR = "test_output"

    # Execute the command as specified in the issue
    subprocess.run([
        "python3", "-m", "batched_lm_generation.automodel_base",
        "--model-name", "gpt2",
        "--dataset", "openai_humaneval",
        "--dataset-split", "test",
        "--output-dir", TEST_DIR,
        "--temperature", "0.2",
        "--batch-size", "100",
        "--completion-limit", "20",
        "--dataset-limit", "2",
        "--max-tokens", "5",
        "--stop", "[]"
    ], check=True)

    # Verify the output directory and file count
    json_gz_files = [f for f in Path(TEST_DIR).iterdir() if f.name.endswith(".json.gz")]
    assert len(json_gz_files) == 2, "Expected exactly 2 .json.gz files in the 'test' directory"

    # Define a mapping function that computes the lengths of each completion
    def compute_length(text):
        return str(len(text))

    # Apply the map_generations function
    map_generations(Path(TEST_DIR), compute_length)
    json_gz_files = [f for f in Path(TEST_DIR).iterdir() if f.name.endswith(".results.json.gz")]
    assert len(json_gz_files) == 2, "Expected exactly 2 .json.gz files in the 'test' directory"

