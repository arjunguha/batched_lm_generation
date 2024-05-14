import subprocess
import pytest
import os

def test_automodel_base_generation():
    # Execute the command as specified in the issue
    subprocess.run([
        "python3", "-m", "batched_lm_generation.automodel_base",
        "--model-name", "gpt2",
        "--dataset", "openai_humaneval",
        "--dataset-split", "test",
        "--output-dir", "test",
        "--temperature", "0.2",
        "--batch-size", "100",
        "--completion-limit", "20",
        "--dataset-limit", "2",
        "--max-tokens", "5",
        "--stop", "[]"
    ], check=True)

    # Verify the output directory and file count
    files = os.listdir('test')
    json_gz_files = [f for f in files if f.endswith('.json.gz')]
    assert len(json_gz_files) == 2, "Expected exactly 2 .json.gz files in the 'test' directory"
